/*
      ____  ____
     /   /\/   /
    /___/  \  /   Copyright (c) 2022, Xilinx®.
    \   \   \/    Author: Víctor Mayoral Vilches <victorma@xilinx.com>
     \   \
     /   /        Licensed under the Apache License, Version 2.0 (the "License");
    /___/   /\    you may not use this file except in compliance with the License.
    \   \  /  \   You may obtain a copy of the License at
     \___\/\___\            http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.

    Inspired by past work from Willow Garage, Inc., Andreas Klintberg,
      Joshua Whitley
*/

#ifndef STEREO_IMAGE_PROC_RESIZE_FPGA_HPP_
#define STEREO_IMAGE_PROC_RESIZE_FPGA_HPP_

#include <rclcpp/rclcpp.hpp>
#include <rclcpp_components/register_node_macro.hpp>
#include <sensor_msgs/image_encodings.hpp>
#include <image_transport/image_transport.hpp>
#include <image_transport/subscriber_filter.hpp>
#include <ament_index_cpp/get_resource.hpp>
#include <image_geometry/stereo_camera_model.h>
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/sync_policies/exact_time.h>
#include <stereo_msgs/msg/disparity_image.hpp>
#include <cstring>
#include <memory>
#include <sstream>
#include <string>
#include <vector>
#include <thread>
#include <mutex>
#include <chrono>

#include <vitis_common/common/ros_opencl_120.hpp>

namespace stereo_image_proc
{

class DisparityNodeFPGA : public rclcpp::Node
{
public:
  explicit DisparityNodeFPGA(const rclcpp::NodeOptions & options);

protected:

  int height_;
  int width_;
  unsigned char bm_state_[4];

  cl::Kernel* krnl_;
  cl::Context* context_;
  cl::CommandQueue* queue_;

  std::mutex connect_mutex_;

private:

  // Subscriptions
  image_transport::SubscriberFilter sub_l_image_, sub_r_image_;
  message_filters::Subscriber<sensor_msgs::msg::CameraInfo> sub_l_info_, sub_r_info_;
using ExactPolicy = message_filters::sync_policies::ExactTime<
    sensor_msgs::msg::Image,
    sensor_msgs::msg::CameraInfo,
    sensor_msgs::msg::Image,
    sensor_msgs::msg::CameraInfo>;
  using ApproximatePolicy = message_filters::sync_policies::ApproximateTime<
    sensor_msgs::msg::Image,
    sensor_msgs::msg::CameraInfo,
    sensor_msgs::msg::Image,
    sensor_msgs::msg::CameraInfo>;
  using ExactSync = message_filters::Synchronizer<ExactPolicy>;
  using ApproximateSync = message_filters::Synchronizer<ApproximatePolicy>;
  std::shared_ptr<ExactSync> exact_sync_;
  std::shared_ptr<ApproximateSync> approximate_sync_;
  // Publications
  std::shared_ptr<rclcpp::Publisher<stereo_msgs::msg::DisparityImage>> pub_disparity_;

  // Handle to parameters callback
  rclcpp::Node::OnSetParametersCallbackHandle::SharedPtr on_set_parameters_callback_handle_;

  // Processing state (note: only safe because we're single-threaded!)
  image_geometry::StereoCameraModel model_;
  // contains scratch buffers for block matching
  // TODO: replace with acceleration kernel
  //stereo_image_proc::StereoProcessor block_matcher_;

  void connectCb();
  void imageCb(
    const sensor_msgs::msg::Image::ConstSharedPtr & l_image_msg,
    const sensor_msgs::msg::CameraInfo::ConstSharedPtr & l_info_msg,
    const sensor_msgs::msg::Image::ConstSharedPtr & r_image_msg,
    const sensor_msgs::msg::CameraInfo::ConstSharedPtr & r_info_msg);

  rcl_interfaces::msg::SetParametersResult parameterSetCb(
    const std::vector<rclcpp::Parameter> & parameters);

};

}  // namespace stereo_image_proc

#endif  // STEREO_IMAGE_PROC_DISPARITY_FPGA_HPP_
