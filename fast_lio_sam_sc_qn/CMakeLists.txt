cmake_minimum_required(VERSION 3.14.4)
project(fast_lio_sam_sc_qn)

### set compiler
set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++17 -pthread -fexceptions -g -ggdb -llz4 ") #c++17, for parallel programming and for Eigen (see https://eigen.tuxfamily.org/dox/group__TopicStructHavingEigenMembers.html)
set(CMAKE_THREAD_PREFER_PTHREAD TRUE)
include(FindOpenMP) #For Nano-GICP, Quatro #The best way to set proper compiler settings for using OpenMP in all platforms
if(OPENMP_FOUND) #For Nano-GICP, Quatro #The best way to set proper compiler settings for using OpenMP in all platforms
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
else(OPENMP_FOUND)
  message("ERROR: OpenMP could not be found.")
endif(OPENMP_FOUND)
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -Wextra ${CMAKE_CXX_FLAGS} -fopenmp -ftree-vectorize -funroll-loops")

add_definitions(-DROOT_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/\")

### get packages
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)

find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_eigen REQUIRED)
find_package(PCL REQUIRED)
find_package(pcl_ros REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(pcl_msgs REQUIRED)
find_package(message_filters REQUIRED)
find_package(teaserpp REQUIRED)
find_package(nano_gicp REQUIRED)
find_package(quatro REQUIRED)

find_package(FMT REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(GTSAM QUIET)
find_package(yaml-cpp REQUIRED)
find_package(rosbag2_cpp REQUIRED)
find_package(rosbag2_storage REQUIRED)
find_package(ament_index_cpp REQUIRED)
find_package(gflags REQUIRED)
find_package(glog REQUIRED)
find_package(OpenCV REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(std_srvs REQUIRED)
find_package(fast_lio REQUIRED)
find_package(livox_ros_driver2 REQUIRED)

if(PCL_FOUND)
  message(WARNING "PCL_VER: ${PCL_VERSION}, and PCL_DIR: ${PCL_INCLUDE_DIRS}")
endif()
if(Eigen3_FOUND)
  message(WARNING "EIGEN_VER: ${EIGEN3_VERSION_STRING}, and Eigen DIR: ${EIGEN3_INCLUDE_DIR}")
endif()
if(GTSAM_FOUND)
  message(WARNING "GTSAM_VER: ${GTSAM_VERSION}, and GTSAM DIR: ${GTSAM_INCLUDE_DIR}")
endif()

### get packages' headers
include_directories(
  include
  ${CMAKE_SOURCE_DIR}/../third_party/scancontext_tro
  ${ament_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
  ${GTSAM_INCLUDE_DIR}
  ${YAML_CPP_INCLUDE_DIRS}
)


set(dependencies
  rclcpp
  ament_index_cpp
  rosbag2_cpp
  rosbag2_storage

  std_msgs
  geometry_msgs
  sensor_msgs
  nav_msgs
  visualization_msgs
  message_filters

  nano_gicp
  quatro

  tf2
  tf2_eigen
  tf2_ros
  tf2_geometry_msgs

  pcl_ros
  pcl_conversions
  pcl_msgs

  glog
  OpenCV
  cv_bridge
  gflags

  GTSAM
  Eigen3
  yaml-cpp

  livox_ros_driver2
  std_srvs
  fast_lio
)


set(common_libraries ${PCL_LIBRARIES} ${EIGEN3_LIBS} ${OpenMP_LIBS} yaml-cpp::yaml-cpp teaserpp::teaser_registration teaserpp::teaser_io gflags glog gtsam)

add_library(${PROJECT_NAME}_modules ${CMAKE_SOURCE_DIR}/../third_party/scancontext_tro/Scancontext.cpp src/loop_closure.cpp src/fast_lio_sam_sc_qn.cpp)
ament_target_dependencies(${PROJECT_NAME}_modules ${dependencies})
target_link_libraries(${PROJECT_NAME}_modules ${common_libraries})

add_executable(${PROJECT_NAME}_node src/main.cpp)
ament_target_dependencies(${PROJECT_NAME}_node ${dependencies})
target_link_libraries(${PROJECT_NAME}_node ${common_libraries} ${PROJECT_NAME}_modules)

install(
  DIRECTORY include/
  DESTINATION include
)

install(TARGETS ${PROJECT_NAME}_modules
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
  INCLUDES DESTINATION include
)

install(
  TARGETS ${PROJECT_NAME}_node
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY launch 
  DESTINATION share/${PROJECT_NAME} 
  FILES_MATCHING PATTERN "*.py"
)

install(DIRECTORY config 
  DESTINATION share/${PROJECT_NAME}
)

ament_export_dependencies(rosidl_default_runtime rclcpp) 
ament_export_include_directories(include)
ament_export_libraries(${PROJECT_NAME}_modules)

ament_package()
