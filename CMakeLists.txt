cmake_minimum_required(VERSION 3.22)

project(RoBoard_Service_Tracking)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(OpenCV REQUIRED)
add_executable(${PROJECT_NAME} src/main.cpp src/vit_loader.c)
target_include_directories(${PROJECT_NAME} SYSTEM PRIVATE ${OpenCV_include_dirs} ..)
target_link_libraries(${PROJECT_NAME} PRIVATE ${OpenCV_LIBRARIES})
