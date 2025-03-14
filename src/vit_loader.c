// Copyright 2023-2024, Collabora, Ltd.
// SPDX-License-Identifier: BSL-1.0
/*!
 * @file
 * @brief  Visual-Inertial Tracking consumer helper.
 * @author Jakob Bornecrantz <jakob@collabora.com>
 * @author Simon Zeni <simon.zeni@collabora.com>
 * @ingroup aux_tracking
 *
 * This file is a slightly modified version of:
 * https://gitlab.freedesktop.org/monado/monado/-/blob/6e4a3a4759457ce75c950bf95ce6ff6e555cd158/src/xrt/auxiliary/tracking/t_vit_loader.c
 *
 */
#include "../include/vit_loader.h"
#include <stdio.h>

#if defined(__linux__) || defined(__ANDROID__)
#include <dlfcn.h>
#endif

static inline bool vit_get_proc(void *handle, const char *name,
                                void *proc_ptr) {
#if defined(__linux__) || defined(__ANDROID__)
  void *proc = dlsym(handle, name);
  char *err = dlerror();
  if (err != NULL) {
    (void)fprintf(stderr, "Failed to load symbol %s\n", err);
    return false;
  }

  *(void **)proc_ptr = proc;
  return true;
#else
#error "Unknown platform"
#endif
}

bool t_vit_bundle_load(struct t_vit_bundle *vit, const char *path) {
#if defined(__linux__) || defined(__ANDROID__)
  vit->handle = dlopen(path, RTLD_LAZY);
  if (vit->handle == NULL) {
    (void)fprintf(stderr, "Failed to open VIT library: %s\n", dlerror());
    return false;
  }
#else
#error "Unknown platform"
#endif

#define GET_PROC(SYM)                                                          \
  do {                                                                         \
    if (!vit_get_proc(vit->handle, "vit_" #SYM, &vit->SYM)) {                  \
      return false;                                                            \
    }                                                                          \
  } while (0)

  // Get the version first.
  GET_PROC(api_get_version);
  vit->api_get_version(&vit->version.major, &vit->version.minor,
                       &vit->version.patch);

  // Check major version.
  if (vit->version.major != VIT_HEADER_VERSION_MAJOR) {
    (void)fprintf(
        stderr, "Incompatible versions: expecting %u.%u.%u but got %u.%u.%u", //
        VIT_HEADER_VERSION_MAJOR, VIT_HEADER_VERSION_MINOR,
        VIT_HEADER_VERSION_PATCH, //
        vit->version.major, vit->version.minor, vit->version.patch);
    dlclose(vit->handle);
    return false;
  }

  GET_PROC(tracker_create);
  GET_PROC(tracker_destroy);
  GET_PROC(tracker_has_image_format);
  GET_PROC(tracker_get_supported_extensions);
  GET_PROC(tracker_get_enabled_extensions);
  GET_PROC(tracker_enable_extension);
  GET_PROC(tracker_start);
  GET_PROC(tracker_stop);
  GET_PROC(tracker_reset);
  GET_PROC(tracker_is_running);
  GET_PROC(tracker_push_imu_sample);
  GET_PROC(tracker_push_img_sample);
  GET_PROC(tracker_add_imu_calibration);
  GET_PROC(tracker_add_camera_calibration);
  GET_PROC(tracker_pop_pose);
  GET_PROC(tracker_get_timing_titles);
  GET_PROC(pose_destroy);
  GET_PROC(pose_get_data);
  GET_PROC(pose_get_timing);
  GET_PROC(pose_get_features);
#undef GET_PROC

  return true;
}

void t_vit_bundle_unload(struct t_vit_bundle *vit) {
#if defined(__linux__) || defined(__ANDROID__)
  dlclose(vit->handle);
#else
#error "Unknown platform"
#endif
}
