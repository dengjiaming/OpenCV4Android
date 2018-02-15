cmake_minimum_required(VERSION 3.4.1)

set(CMAKE_VERBOSE_MAKEFILE on)
set(libs "${CMAKE_SOURCE_DIR}/src/main/libs")
include_directories(${CMAKE_SOURCE_DIR}/src/main/cpp/include)

add_library(libopencv_java3 SHARED IMPORTED )
set_target_properties(libopencv_java3 PROPERTIES
    IMPORTED_LOCATION "${libs}/${ANDROID_ABI}/libopencv_java3.so")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=gnu++11 -fexceptions -frtti")

add_library(native-lib
            SHARED
            src/main/cpp/native-lib.cpp)

add_library(detection_based_tracker
            SHARED
            src/main/cpp/DetectionBasedTracker_jni.cpp
            src/main/cpp/lbph_faces.cpp
            src/main/cpp/eigen_faces.cpp
            src/main/cpp/fisher_faces.cpp
            src/main/cpp/facerec.cpp
            src/main/cpp/face_basic.cpp
            src/main/cpp/predict_collector.cpp)

find_library(log-lib
              log )

target_link_libraries(native-lib android log
    libopencv_java3 #used for java sdk
    ${log-lib})

target_link_libraries(detection_based_tracker android log
    libopencv_java3 #used for java sdk
    ${log-lib})