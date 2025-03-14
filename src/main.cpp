// Copyright 2024, Technical University of Munich.
// SPDX-License-Identifier: BSL-1.0
/*!
 * @file
 * @brief  Visual-Intertial Tracking tracker C++ helper.
 * @author Mateo de Mayo <mateo.demayo@tum.de>
 *
 *
 * Some of the functionality in this file is based in the euroc_player.cpp file
 * from Monado
 */

#include "../include/vit_loader.h"
#include "../lib/vit/vit_implementation_helper.hpp"
#include "../lib/vit/vit_interface.h"

#include <opencv2/opencv.hpp>

#include <array>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <future>
#include <string>
#include <thread>
#include <vector>

using namespace vit;
using std::array;
using std::future;
using std::ifstream;
using std::launch;
using std::ofstream;
using std::pair;
using std::string;
using std::to_string;
using std::vector;
using std::chrono::nanoseconds;
using std::this_thread::sleep_for;

using Timestamp = int64_t;
using ImgEntry =
    pair<Timestamp, string>; // Not using ImgSample to avoid loading full images
using ImuSamples = vector<ImuSample>;
using ImgSamples = vector<ImgEntry>;
using PoseSamples = vector<PoseData>;
using StreamThread = future<void>;

constexpr bool REALTIME_STREAM =
    false; // If false, push samples as quick as possible
constexpr bool SHOW_STREAMED_IMAGES =
    false; // If true, do cv::imshow of the cam0 images

#define ASSERT(cond, ...)                                                      \
  /* NOLINTBEGIN */                                                            \
  do {                                                                         \
    if (!(cond)) {                                                             \
      printf("Assertion failed @%s:%d\n", __func__, __LINE__);                 \
      printf(__VA_ARGS__);                                                     \
      printf("\n");                                                            \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (false);                                                             \
  /* NOLINTEND */

#define ASSERT_(cond) ASSERT(cond, "%s", #cond);
#define PRINT(...) printf(__VA_ARGS__); // NOLINT
#define DO_(statement)                                                         \
  _res_DO = statement;                                                         \
  ASSERT(_res_DO == VIT_SUCCESS, "VIT failure code: %d", _res_DO)
#define DO(statement)                                                          \
  Result _res_DO = VIT_SUCCESS;                                                \
  DO_(statement)

class VITTracker {
  t_vit_bundle vit{};
  struct vit_tracker *tracker{};
  StreamThread printer_thread{};
  Timestamp last_ts = -1; // Last expected frame timestamp from the dataset

  void pose_printer() {
    constexpr auto POLL_SLEEP = nanoseconds(50'000'000);
    constexpr int CSV_PRECISION = 10;
    string CSV_EOL = "\r\n";

    ofstream file{"estimate.csv"};
    file << "#timestamp [ns], px [m], py [m], pz [m], qw, qx, qy, qz"
         << CSV_EOL;
    file << std::fixed << std::setprecision(CSV_PRECISION);

    PoseData pose{};
    while (pose.timestamp != last_ts) {
      sleep_for(POLL_SLEEP);
      while (pop_pose(pose)) {
        file << pose.timestamp << ",";
        file << pose.px << "," << pose.py << "," << pose.pz << ",";
        file << pose.ow << "," << pose.ox << "," << pose.oy << "," << pose.oz
             << CSV_EOL;
      }
    }
  }

public:
  VITTracker(const string &libvit_path, const Config &config, Timestamp last_ts)
      : last_ts(last_ts) {
    bool success = t_vit_bundle_load(&vit, libvit_path.c_str());
    ASSERT(success, "Failed to load VIT system library from '%s'",
           libvit_path.c_str());
    DO(vit.tracker_create(&config, &tracker));
  }

  ~VITTracker() { vit.tracker_destroy(tracker); }

  void start() {
    DO(vit.tracker_start(tracker));
    printer_thread = async(launch::async, &VITTracker::pose_printer, this);
  }

  void push_imu(const ImuSample &sample) {
    DO(vit.tracker_push_imu_sample(tracker, &sample));
  }

  void push_img(const ImgSample &sample) {
    DO(vit.tracker_push_img_sample(tracker, &sample));
  }

  void stop() {
    printer_thread.get();
    DO(vit.tracker_stop(tracker));
  }

  bool is_running() {
    bool is_running = false;
    DO(vit.tracker_is_running(tracker, &is_running));
    return is_running;
  }

  bool pop_pose(PoseData &out_pose) {
    Pose *pose = nullptr;
    DO(vit.tracker_pop_pose(tracker, &pose));

    if (!pose)
      return false;

    DO_(vit.pose_get_data(pose, &out_pose));
    vit.pose_destroy(pose);

    return true;
  }
};

class EurocDataset {
  string dataset_path;
  int imu_count = 1; // NOTE: This code assumes there is exactly one IMU
  int cam_count = 0;
  bool is_colored = false;
  bool has_imu = false;
  bool has_gt = false;
  int width = 0;
  int height = 0;
  Timestamp last_cam_ts = -1;

  PoseSamples gts;
  ImuSamples imus;
  vector<ImgSamples> cams;

  StreamThread imus_stream;
  StreamThread cams_stream;

  //! Parse and load all IMU samples into `samples`, assumes data.csv is well
  //! formed If `read_n` > 0, read at most that amount of samples Returns
  //! whether the appropriate data.csv file could be opened
  bool load_imu_data(ImuSamples &samples, int64_t read_n = -1) {
    string csv_filename = dataset_path + "/mav0/imu0/data.csv";
    ifstream fin{csv_filename};
    if (!fin.is_open()) {
      return false;
    }

    // EuRoC imu columns: ts wx wy wz ax ay az
    constexpr int COLUMN_COUNT = 6;       // wx wy wz ax ay az
    constexpr int AX = 3, AY = 4, AZ = 5; // accelerometer indices
    constexpr int WX = 0, WY = 1, WZ = 2; // gyroscope indices

    string line;
    getline(fin, line); // Skip header line

    while (getline(fin, line) && read_n-- != 0) {
      Timestamp t = -1;
      array<float, COLUMN_COUNT> v{};
      size_t i = 0;
      size_t j = line.find(',');
      t = stoll(line.substr(i, j));
      for (int k = 0; k < COLUMN_COUNT; k++) {
        i = j;
        j = line.find(',', i + 1);
        v.at(k) = stof(line.substr(i + 1, j));
      }

      ImuSample sample{t, v[AX], v[AY], v[AZ], v[WX], v[WY], v[WZ]};
      samples.push_back(sample);
    }
    return true;
  }

  //! Parse and load image names and timestamps into `samples`
  //! If read_n > 0, read at most that amount of samples
  //! Returns whether the appropriate data.csv file could be opened
  bool load_cam_data(ImgSamples &samples, size_t cam_id, int64_t read_n = -1) {
    // Parse image data, assumes data.csv is well formed
    string cam_name = "cam" + to_string(cam_id);
    string imgs_path = dataset_path + "/mav0/" + cam_name + "/data";
    string csv_filename = dataset_path + "/mav0/" + cam_name + "/data.csv";
    ifstream fin{csv_filename};
    if (!fin.is_open()) {
      return false;
    }

    string line;
    getline(fin, line); // Skip header line
    while (getline(fin, line) && read_n-- != 0) {
      size_t i = line.find(',');
      Timestamp timestamp = stoll(line.substr(0, i));
      string img_name_tail = line.substr(i + 1);

      // Standard euroc datasets use CRLF line endings, so let's remove the
      // extra '\r'
      if (img_name_tail.back() == '\r') {
        img_name_tail.pop_back();
      }

      string img_name =
          imgs_path + "/" +
          img_name_tail; // NOLINT(performance-inefficient-string-concatenation)
      ImgEntry sample{timestamp, img_name};
      samples.push_back(sample);
    }
    return true;
  }

  //! Parse and load ground truth (gt) poses into `samples` from specified gt
  //! device name If read_n > 0, read at most that amount of samples Returns
  //! whether the appropriate data.csv file could be opened
  bool load_gt_data(string &gtdev, PoseSamples &samples, int64_t read_n = -1) {
    vector<string> gt_devices = {"gt", "state_groundtruth_estimate0", "vicon0",
                                 "mocap0", "leica0"};
    if (gtdev != "") {
      gt_devices.insert(gt_devices.begin(), gtdev);
    }

    ifstream fin;
    for (string &device : gt_devices) {
      string csv_filename = dataset_path + "/mav0/" + device + "/data.csv";
      fin = ifstream{csv_filename};
      if (fin.is_open()) {
        gtdev = device;
        break;
      }
    }

    if (!fin.is_open()) {
      return false;
    }

    // EuRoC groundtruth columns: ts px py pz qw qx qy qz
    constexpr int COLUMN_COUNT = 7; // px py pz qw qx qy qz
    constexpr int PX = 0, PY = 1, PZ = 2;
    constexpr int QX = 4, QY = 5, QZ = 6, QW = 3;

    string line;
    getline(fin, line); // Skip header line

    while (getline(fin, line) && read_n-- != 0) {
      Timestamp t = -1;
      array<float, COLUMN_COUNT> v = {
          0, 0, 0, 1, 0, 0, 0}; // Set identity orientation for leica0
      size_t i = 0;
      size_t j = line.find(',');
      t = stoll(line.substr(i, j));
      for (size_t k = 0; k < COLUMN_COUNT && j != string::npos; k++) {
        i = j;
        j = line.find(',', i + 1);
        v.at(k) = stof(line.substr(i + 1, j));
      }

      PoseData pose{t,     v[PX], v[PY], v[PZ], v[QX], v[QY],
                    v[QZ], v[QW], 0,     0,     0};
      samples.push_back(pose);
    }
    return true;
  }

  void load() {
    string gtdev = "";
    gts.clear();
    imus.clear();
    for (auto &imgs : cams) {
      imgs.clear();
    }

    load_gt_data(gtdev, gts);
    load_imu_data(imus);
    for (int i = 0; i < cam_count; i++) {
      load_cam_data(cams.at(i), i);
    }
  }

  //! Determine and fill attributes of the dataset pointed by `path`
  //! Assertion fails if `path` does not point to an euroc dataset
  void prefill_dataset_info() {
    ImgSamples samples;
    ImuSamples _1;
    PoseSamples _2;
    string gtdev = "";

    size_t i = 0;
    bool has_camera = load_cam_data(samples, i, 1);
    while ((has_camera = load_cam_data(samples, ++i, 0))) {
      continue;
    }

    size_t nof_cameras = i;
    bool imu_found = load_imu_data(_1, 0);
    bool gt_found = load_gt_data(gtdev, _2, 0);
    bool is_valid_dataset = nof_cameras > 0 && imu_found;
    ASSERT(is_valid_dataset, "Invalid dataset %s", dataset_path.c_str());

    cv::Mat first_cam0_img = cv::imread(samples[0].second, cv::IMREAD_ANYCOLOR);
    cam_count = (int)nof_cameras;
    is_colored = first_cam0_img.channels() == 3;
    has_gt = gt_found;
    has_imu = imu_found;
    width = first_cam0_img.cols;
    height = first_cam0_img.rows;
  }

  void stream_imu(VITTracker &tracker) {
    for (int i = 0; i < imus.size() - 1; i++) {
      ImuSample curr = imus.at(i);
      ImuSample next = imus.at(i + 1);
      tracker.push_imu(curr);
      if constexpr (REALTIME_STREAM)
        sleep_for(nanoseconds(next.timestamp - curr.timestamp));
    }
    tracker.push_imu(imus.back());
  }

  void stream_cams(VITTracker &tracker) {
    vector<cv::Mat> frameset_mats(cam_count);

    auto frameset_from_entry = [this, &frameset_mats](int i) {
      vector<ImgSample> frameset(cam_count);

      for (int cam_id = 0; cam_id < cam_count; cam_id++) {
        const ImgSamples &imgs = cams.at(cam_id);
        auto [ts, path] = imgs.at(i);
        cv::Mat img = cv::imread(path, cv::IMREAD_ANYCOLOR);
        ASSERT_(img.type() == CV_8UC1);

        ImgSample sample{};
        sample.cam_index = cam_id;
        sample.timestamp = ts;
        sample.data = img.ptr<uint8_t>();
        sample.width = img.cols;
        sample.height = img.rows;
        sample.stride = img.step[0];
        sample.size = img.step[0] * img.rows;
        sample.format = VIT_IMAGE_FORMAT_L8;

        frameset.at(cam_id) = sample;
        frameset_mats.at(cam_id) = img; // To hold the cv::Mat in memory
      }

      return frameset;
    };

    size_t img_count = cams.at(0).size();
    vector<ImgSample> curr = frameset_from_entry(0);
    for (int i = 0; i < img_count - 1; i++) {
      for (ImgSample &frame : curr)
        tracker.push_img(frame);

      if constexpr (SHOW_STREAMED_IMAGES) {
        cv::imshow("cam0", frameset_mats.at(0));
        cv::waitKey(4);
      }

      vector<ImgSample> next = frameset_from_entry(i + 1);
      if constexpr (REALTIME_STREAM)
        sleep_for(nanoseconds(next.at(0).timestamp - curr.at(0).timestamp));
      curr = next;
    }
    for (ImgSample &frame : curr)
      tracker.push_img(frame);
  }

public:
  EurocDataset(const string &path) : dataset_path(path) {
    prefill_dataset_info();
    PRINT("Dataset path: '%s'\n", dataset_path.c_str());
    PRINT("\tcam_count=%d\n", cam_count);
    PRINT("\tis_colored=%d\n", is_colored);
    PRINT("\thas_imu=%d\n", has_imu);
    PRINT("\thas_gt=%d\n", has_gt);
    PRINT("\twidth=%d\n", width);
    PRINT("\theight=%d\n", height);

    ASSERT_(cam_count > 0);
    cams.resize(cam_count);

    load();
  }

  void stream(VITTracker &tracker) {
    imus_stream = async(launch::async, [&] { stream_imu(tracker); });
    cams_stream = async(launch::async, [&] { stream_cams(tracker); });
  }

  void wait_stream_end() {
    cams_stream.get();
    imus_stream.get();
  }

  uint32_t get_cam_count() { return cam_count; }
  uint32_t get_imu_count() { return imu_count; }
  Timestamp get_last_timestamp() { return cams.at(0).back().first; }
};

int main(int argc, char **argv) {

  cv::VideoCapture.open(0);

  cv::Mat frame;

  cap >> frame;

  cv::imshow("Flux", frame);

  sleep_for(std::chrono::seconds(10));

  ASSERT(argc == 4, "Invalid arguments. Usage: ./vit_consumer_example "
                    "<DATASET> <LIBRARY> <CONFIG>");
  string dataset_path = argv[1];
  string library_path = argv[2];
  string config_path = argv[3];
  bool show_ui = true;

  EurocDataset dataset{dataset_path};
  Config config{config_path.c_str(), dataset.get_cam_count(),
                dataset.get_imu_count(), show_ui};
  VITTracker tracker{library_path, config, dataset.get_last_timestamp()};

  tracker.start();
  dataset.stream(tracker);
  dataset.wait_stream_end();
  tracker.stop();
}
