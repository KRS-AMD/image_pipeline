// Copyright (c) 2008, Willow Garage, Inc.
// All rights reserved.
//
// Software License Agreement (BSD License 2.0)
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above
//    copyright notice, this list of conditions and the following
//    disclaimer in the documentation and/or other materials provided
//    with the distribution.
//  * Neither the name of the copyright holder nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
// FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
// COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
// INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
// BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
// LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
// ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#include <memory>
#include <mutex>
#include <vector>
#include <chrono>

#include <cv_bridge/cv_bridge.h>
#include <image_geometry/stereo_camera_model.h>
#include <image_transport/image_transport.hpp>
#include <image_transport/subscriber_filter.hpp>
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/sync_policies/exact_time.h>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_components/register_node_macro.hpp>
#include <sensor_msgs/image_encodings.hpp>
#include <stereo_msgs/msg/disparity_image.hpp>

#include <opencv2/calib3d/calib3d.hpp>

#include "stereo_image_proc/disparity_fpga.hpp"
#include "xf_stereo_pipeline_config.h"
#include <vitis_common/common/xf_headers.hpp>
#include <vitis_common/common/utilities.hpp>
#include "tracetools_image_pipeline/tracetools.h"

// Forward declaration of utility functions included at the end of this file
std::vector<cl::Device> get_xilinx_devices();
char* read_binary_file(const std::string &xclbin_file_name, unsigned &nb);

namespace stereo_image_proc
{

// Some helper functions for adding a parameter to a collection
static void add_param_to_map(
  std::map<std::string, std::pair<int, rcl_interfaces::msg::ParameterDescriptor>> & parameters,
  const std::string & name,
  const std::string & description,
  const int default_value,
  const int from_value,
  const int to_value,
  const int step)
{
  rcl_interfaces::msg::IntegerRange integer_range;
  integer_range.from_value = from_value;
  integer_range.to_value = to_value;
  integer_range.step = step;
  rcl_interfaces::msg::ParameterDescriptor descriptor;
  descriptor.description = description;
  descriptor.integer_range = {integer_range};
  parameters[name] = std::make_pair(default_value, descriptor);
}

static void add_param_to_map(
  std::map<std::string, std::pair<double, rcl_interfaces::msg::ParameterDescriptor>> & parameters,
  const std::string & name,
  const std::string & description,
  const double default_value,
  const double from_value,
  const double to_value,
  const double step)
{
  rcl_interfaces::msg::FloatingPointRange floating_point_range;
  floating_point_range.from_value = from_value;
  floating_point_range.to_value = to_value;
  floating_point_range.step = step;
  rcl_interfaces::msg::ParameterDescriptor descriptor;
  descriptor.description = description;
  descriptor.floating_point_range = {floating_point_range};
  parameters[name] = std::make_pair(default_value, descriptor);
}

DisparityNodeFPGA::DisparityNodeFPGA(const rclcpp::NodeOptions & options)
: rclcpp::Node("DisparityNodeFPGA", options)
{
  using namespace std::placeholders;

  // Xilinx init

  cl_int err;
  unsigned fileBufSize;

  // Get the device:
  std::vector<cl::Device> devices = get_xilinx_devices();
  // devices.resize(1);  // done below
  cl::Device device = devices[0];

  // Context, command queue and device name:
  OCL_CHECK(err, context_ = new cl::Context(device, NULL, NULL, NULL, &err));
  OCL_CHECK(err, queue_ = new cl::CommandQueue(*context_, device,
                                    CL_QUEUE_PROFILING_ENABLE, &err));
  OCL_CHECK(err, std::string device_name =
                                  device.getInfo<CL_DEVICE_NAME>(&err));

  std::cout << "INFO: Device found - " << device_name << std::endl;

  // Load binary:
  // NOTE: hardcoded path according to dfx-mgrd conventions
  // TODO: generalize this using launch extra_args for composable Nodes
  // see https://github.com/ros2/launch_ros/blob/master/launch_ros/launch_ros/descriptions/composable_node.py#L45
  char* fileBuf = read_binary_file(
        "/lib/firmware/xilinx/stereo_image_proc/stereo_image_proc.xclbin",
        fileBufSize);
  cl::Program::Binaries bins{{fileBuf, fileBufSize}};
  devices.resize(1);
  OCL_CHECK(err, cl::Program program(*context_, devices, bins, NULL, &err));

  // Create a kernel:
  OCL_CHECK(err, krnl_ = new cl::Kernel(program, "stereolbm_accel", &err));

  // End Xilinx init

  // ROS Declare/read parameters
  int queue_size = this->declare_parameter("queue_size", 5);
  bool approx = this->declare_parameter("approximate_sync", false);
  this->declare_parameter("use_system_default_qos", false);

  // Synchronize callbacks
  if (approx) {
    approximate_sync_.reset(
      new ApproximateSync(
        ApproximatePolicy(queue_size),
        sub_l_image_, sub_l_info_,
        sub_r_image_, sub_r_info_));
    approximate_sync_->registerCallback(
      std::bind(&DisparityNodeFPGA::imageCb, this, _1, _2, _3, _4));
  } else {
    exact_sync_.reset(
      new ExactSync(
        ExactPolicy(queue_size),
        sub_l_image_, sub_l_info_,
        sub_r_image_, sub_r_info_));
    exact_sync_->registerCallback(
      std::bind(&DisparityNodeFPGA::imageCb, this, _1, _2, _3, _4));
  }

  // Register a callback for when parameters are set
  on_set_parameters_callback_handle_ = this->add_on_set_parameters_callback(
    std::bind(&DisparityNodeFPGA::parameterSetCb, this, _1));

  // Describe int parameters
  std::map<std::string, std::pair<int, rcl_interfaces::msg::ParameterDescriptor>> int_params;
  add_param_to_map(
    int_params,
    "prefilter_cap",
    "Bound on normalized pixel values",
    31, 1, 63, 1);
  add_param_to_map(
    int_params,
    "min_disparity",
    "Disparity to begin search at in pixels",
    0, -2048, 2048, 1);
  add_param_to_map(
    int_params,
    "texture_threshold",
    "Filter out if SAD window response does not exceed texture threshold",
    10, 0, 10000, 1);

  // Double params
  std::map<std::string, std::pair<double, rcl_interfaces::msg::ParameterDescriptor>> double_params;
  add_param_to_map(
    double_params,
    "uniqueness_ratio",
    "Filter out if best match does not sufficiently exceed the next-best match",
    15.0, 0.0, 100.0, 0.0);

  // Declaring parameters triggers the previously registered callback

  this->declare_parameters("", int_params);
  this->declare_parameters("", double_params);

 // Get params and assign values to to character sequence for stereolbm_accel block matching state
  std::vector<std::string> param_names = {"prefilter_cap", "min_disparity", "texture_threshold", "uniqueness_ratio"};

  std::vector<rclcpp::Parameter> params = this->get_parameters(param_names);
  for (auto &param : params)
  {
    std::string param_name = param.get_name().c_str();
    std::string param_val = param.value_to_string().c_str();

    if (param_name == "prefilter_cap") {
    bm_state_[0] = static_cast<unsigned char>(std::stoi(param_val, nullptr, 10));
    }
    else if (param_name == "uniqueness_ratio") {
    bm_state_[1] = static_cast<unsigned char>(std::stoi(param_val, nullptr, 10));
    }
    else if (param_name == "texture_threshold") {
    bm_state_[2] = static_cast<unsigned char>(std::stoi(param_val, nullptr, 10));
    }
    else if (param_name ==  "min_disparity") {
    bm_state_[3] = static_cast<unsigned char>(std::stoi(param_val, nullptr, 10));
    }
    else {
      continue;
    } // end-if-else-if

  } // end-for

  pub_disparity_= this->create_publisher<stereo_msgs::msg::DisparityImage>("disparity", 1);

  connectCb();
}

// Handles (un)subscribing when clients (un)subscribe
void DisparityNodeFPGA::connectCb()
{
  // TODO(jacobperron): Add unsubscribe logic when we use graph events
  image_transport::TransportHints hints(this, "raw");
  const bool use_system_default_qos = this->get_parameter("use_system_default_qos").as_bool();
  rclcpp::QoS image_sub_qos = rclcpp::SensorDataQoS();
  if (use_system_default_qos) {
    image_sub_qos = rclcpp::SystemDefaultsQoS();
  }
  const auto image_sub_rmw_qos = image_sub_qos.get_rmw_qos_profile();
  sub_l_image_.subscribe(this, "left/image_rect", hints.getTransport(), image_sub_rmw_qos);
  sub_l_info_.subscribe(this, "left/camera_info", image_sub_rmw_qos);
  sub_r_image_.subscribe(this, "right/image_rect", hints.getTransport(), image_sub_rmw_qos);
  sub_r_info_.subscribe(this, "right/camera_info", image_sub_rmw_qos);
}

void DisparityNodeFPGA::imageCb(
  const sensor_msgs::msg::Image::ConstSharedPtr & l_image_msg,
  const sensor_msgs::msg::CameraInfo::ConstSharedPtr & l_info_msg,
  const sensor_msgs::msg::Image::ConstSharedPtr & r_image_msg,
  const sensor_msgs::msg::CameraInfo::ConstSharedPtr & r_info_msg)
{
  // If there are no subscriptions for the disparity image, do nothing
  if (pub_disparity_->get_subscription_count() == 0u) {
    return;
  }

  // Update the camera model
  model_.fromCameraInfo(l_info_msg, r_info_msg);

  // Allocate new disparity image message
  auto disp_msg = std::make_shared<stereo_msgs::msg::DisparityImage>();
  disp_msg->header = l_info_msg->header;
  disp_msg->image.header = l_info_msg->header;

  // Allocate a new disparity info message
  sensor_msgs::msg::CameraInfo::ConstSharedPtr disp_info_msg = std::make_shared<sensor_msgs::msg::CameraInfo>(*l_info_msg);

  // Vitis Vision library ROS to CV image
  cv_bridge::CvImagePtr cv_ptr_l;
  cv_bridge::CvImagePtr cv_ptr_r;
  cv::Mat hls_disp16;
  static const double inv_dpp = 1.0 / NO_OF_DISPARITIES;

  try {
  cv_ptr_l = cv_bridge::toCvCopy(l_image_msg, sensor_msgs::image_encodings::MONO8);
  cv_ptr_r = cv_bridge::toCvCopy(r_image_msg, sensor_msgs::image_encodings::MONO8);
  } catch (cv_bridge::Exception & e) {
	  RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
  /*        TRACEPOINT(stereo_image_proc_disparity_cv,
	  static_cast<const void *>(this),
	  static_cast<const void *>(&(*l_image_msg)),
	  static_cast<const void *>(&(*r_image_msg)));*/
    return;
  }

  // OpenCL section WIP

  cl_int err;
  size_t image_l_in_size_bytes, image_r_in_size_bytes, image_out_size_bytes;

  // Assume MONO (1 channels)
  hls_disp16.create(cv::Size(l_info_msg->width,
                                l_info_msg->height), CV_16UC1);

  image_l_in_size_bytes = l_info_msg->height * l_info_msg->width *
                                  1 * sizeof(unsigned char);

  // Assume same size as left image for the buffer allocation
  image_r_in_size_bytes = image_l_in_size_bytes;

  image_out_size_bytes = l_info_msg->height * l_info_msg->width *
	                          1 * sizeof(unsigned char);

  // Allocate the buffers:
  OCL_CHECK(err, cl::Buffer imageLToDevice(*context_, CL_MEM_READ_ONLY,
                                          image_l_in_size_bytes, NULL, &err));
  OCL_CHECK(err, cl::Buffer imageRToDevice(*context_, CL_MEM_READ_ONLY,
                                          image_r_in_size_bytes, NULL, &err));
  OCL_CHECK(err, cl::Buffer imageFromDevice(*context_, CL_MEM_WRITE_ONLY,
                                            image_out_size_bytes, NULL, &err));
  // Set the kernel arguments
  OCL_CHECK(err, err = krnl_->setArg(0, imageLToDevice));
  OCL_CHECK(err, err = krnl_->setArg(1, imageRToDevice));
  OCL_CHECK(err, err = krnl_->setArg(3, imageFromDevice));
  OCL_CHECK(err, err = krnl_->setArg(2, bm_state_));
  OCL_CHECK(err, err = krnl_->setArg(4, disp_info_msg->width));
  OCL_CHECK(err, err = krnl_->setArg(5, disp_info_msg->height));

  OCL_CHECK(err, queue_->enqueueWriteBuffer(imageRToDevice, CL_TRUE, 0, image_r_in_size_bytes, cv_ptr_r->image.data));

  OCL_CHECK(err, queue_->enqueueWriteBuffer(imageLToDevice, CL_TRUE, 0, image_l_in_size_bytes, cv_ptr_l->image.data));

  cl::Event event_sp;
  OCL_CHECK(err, err = queue_->enqueueTask(*krnl_, NULL, &event_sp));

  OCL_CHECK(err, queue_->enqueueReadBuffer(imageFromDevice, CL_TRUE, 0, image_out_size_bytes, hls_disp16.data));

  // Output cv_bridge image
  cv_bridge::CvImage hls_cv;
  hls_cv.header = cv_ptr_l->header;
  hls_cv.encoding = cv_ptr_l->encoding;
  hls_cv.image = cv::Mat{
	static_cast<int>(disp_info_msg->height),
	static_cast<int>(disp_info_msg->width),
        CV_16UC1,
        hls_disp16.data
    };

  queue_->finish();

  // Fill in DisparityImage image data, data struct used to store image data before converting to 32-bit float
  sensor_msgs::msg::Image & dimage = disp_msg->image;
  dimage.height = disp_info_msg->height;
  dimage.width = disp_info_msg->width;
  dimage.encoding = sensor_msgs::image_encodings::TYPE_32FC1;
  dimage.step = dimage.width * sizeof(float);
  dimage.data.resize(dimage.step * dimage.height);

  // CV mat to convert hls disparity 16-bit to float, dimage data structure
  cv::Mat_<float> dmat(
    dimage.height, dimage.width, reinterpret_cast<float *>(&hls_disp16.data[0]), dimage.step);

  // We convert from 16-bit fixed-point to float disparity and also adjust for any x-offset between
  // the principal points: d = d_fp*inv_dpp - (cx_l - cx_r)
  hls_cv.image.convertTo(dmat, dmat.type(), inv_dpp);//TODO: , -(model.left().cx() - model.right().cx()));
  RCUTILS_ASSERT(dmat.data == &hls_disp16.data[0]);
  // TODO(unknown): is_bigendian?

  // Convert to image message
  hls_cv.toImageMsg(disp_msg->image);

  // End OpenCL
  pub_disparity_->publish(*disp_msg);

} // end cb

}  // namespace stereo_image_proc

// Register component
RCLCPP_COMPONENTS_REGISTER_NODE(stereo_image_proc::DisparityNodeFPGA)
