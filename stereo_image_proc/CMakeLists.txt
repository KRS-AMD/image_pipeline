cmake_minimum_required(VERSION 3.5)
project(stereo_image_proc)

# ROS2 flags
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# LTTng flame chart and graphs
# see https://archive.eclipse.org/tracecompass/doc/stable/org.eclipse.tracecompass.doc.user/LTTng-UST-Analyses.html#Flame_Chart_View
set(CMAKE_CXX_FLAGS "-g -O2 -finstrument-functions")

find_package(ament_cmake_auto REQUIRED)
ament_auto_find_build_dependencies()

find_package(ament_vitis)
find_package(vitis_common REQUIRED)
find_package(OpenCL REQUIRED)
find_package(OpenCV REQUIRED) 
find_package(XRT REQUIRED)  # needs to be reviewed, issues with sysroot isolation
find_package(image_proc)
if(OpenCV_VERSION VERSION_LESS "3.2.0")
  message(FATAL "Minimum OpenCV version is 3.2.0 (found version ${OpenCV_VERSION})")
endif()

# See note in image_proc/CMakeLists.txt
# add_definitions(-DOPENCV_TRAITS_ENABLE_DEPRECATED)

ament_auto_add_library(${PROJECT_NAME} SHARED
  src/${PROJECT_NAME}/stereo_processor.cpp
  src/${PROJECT_NAME}/disparity_node.cpp
  src/${PROJECT_NAME}/point_cloud_node.cpp
)
target_link_libraries(${PROJECT_NAME}
  ${OpenCV_LIBRARIES}
)

rclcpp_components_register_node(${PROJECT_NAME}
  PLUGIN "stereo_image_proc::DisparityNode"
  EXECUTABLE disparity_node
)
rclcpp_components_register_node(${PROJECT_NAME}
  PLUGIN "stereo_image_proc::PointCloudNode"
  EXECUTABLE point_cloud_node
)

# disparity_fpga component (with OpenCL)
ament_auto_add_library(disparity_fpga SHARED src/stereo_image_proc/disparity_fpga.cpp)
target_include_directories(disparity_fpga PUBLIC
  # $<INSTALL_INTERFACE:${CMAKE_INSTALL_PREFIX}/include/vitis_common>
  $ENV{XILINX_HLS}/common/technology/autopilot
  $ENV{XILINX_HLS}/include)
target_link_libraries(disparity_fpga ${OpenCL_LIBRARY})
target_compile_definitions(disparity_fpga PRIVATE "COMPOSITION_BUILDING_DLL")
rclcpp_components_register_nodes(disparity_fpga "stereo_image_proc::DisparityNodeFPGA")
set(node_plugins "${node_plugins}stereo_image_proc::DisparityNodeFPGA;$<TARGET_FILE:disparity_fpga>\n")

# point_cloud_fpga component (with OpenCL)
#ament_auto_add_library(pointcloud_fpga SHARED src/pointcloud_fpga.cpp)
#target_include_directories(pointcloud_fpga PUBLIC
#  # $<INSTALL_INTERFACE:${CMAKE_INSTALL_PREFIX}/include/vitis_common>
#  $ENV{XILINX_HLS}/common/technology/autopilot
#  $ENV{XILINX_HLS}/include)
#target_link_libraries(pointcloud_fpga ${OpenCL_LIBRARY})
#target_compile_definitions(pointcloud_fpga PRIVATE "COMPOSITION_BUILDING_DLL")
#rclcpp_components_register_nodes(pointcloud_fpga "image_proc::PointCloudNodeFPGA")
#set(node_plugins "${node_plugins}image_proc::PointCloudNodeFPGA;$<TARGET_FILE:pointcloud_fpga>\n")

 if(ROS_ACCELERATION)
   # resize_accel kernel
   vitis_acceleration_kernel(
     NAME stereolbm_accel
     FILE src/stereo_image_proc/xf_stereolbm_accel.cpp
     CONFIG cfg/kv260.cfg
     INCLUDE
       include/stereo_image_proc
       ${CMAKE_INSTALL_PREFIX}/include
     TYPE hw
     CLOCK 100000000:stereolbm_accel
     LINK
     # PACKAGE
   )

   # rectify_accel kernel
   #   vitis_acceleration_kernel(
   #  NAME rectify_accel
   #  FILE src/image_proc/xf_rectify_accel.cpp
   #  CONFIG cfg/kv260.cfg
   #  INCLUDE
   #    include/image_proc
   #    ${CMAKE_INSTALL_PREFIX}/include
   #  TYPE hw
   #  CLOCK 100000000:rectify_accel
   #  LINK
   #  # PACKAGE
   #)
 endif()

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()

  set(_PYTHON_EXECUTABLE "${PYTHON_EXECUTABLE}")
  if(WIN32 AND CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(PYTHON_EXECUTABLE "${PYTHON_EXECUTABLE_DEBUG}")
  endif()

  # Test DisparityNode in launch test
  add_launch_test("test/test_disparity_node.py"
    PYTHON_EXECUTABLE "${PYTHON_EXECUTABLE}"
  )
  # TODO(jacobperron): As of Eloquent, they're just plain pytests
  # ament_add_pytest_test("test_disparity_node" test/test_disparity_node.py
  #   PYTHON_EXECUTABLE "${PYTHON_EXECUTABLE}"
  # )

  # Test PointCloudNode in launch test
  add_launch_test("test/test_point_cloud_node.py"
    PYTHON_EXECUTABLE "${PYTHON_EXECUTABLE}"
  )
  # ament_add_pytest_test("test_point_cloud_node" test/test_point_cloud_node.py
  #   PYTHON_EXECUTABLE "${PYTHON_EXECUTABLE}"
  # )

  set(PYTHON_EXECUTABLE "${_PYTHON_EXECUTABLE}")
endif()

ament_auto_package(INSTALL_TO_SHARE launch)
