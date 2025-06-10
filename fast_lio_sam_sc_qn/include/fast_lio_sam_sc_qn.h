#ifndef FAST_LIO_SAM_SC_QN_MAIN_H
#define FAST_LIO_SAM_SC_QN_MAIN_H

///// common headers
#include <ctime>
#include <cmath>
#include <chrono> //time check
#include <vector>
#include <memory>
#include <deque>
#include <mutex>
#include <string>
#include <utility> // pair, make_pair
#include <tuple>
#include <filesystem>
#include <fstream>
#include <iostream>
///// ROS
#include <ament_index_cpp/get_package_share_directory.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rosbag2_cpp/converter_interfaces/serialization_format_converter.hpp>
#include <rosbag2_cpp/reader.hpp>
#include <rosbag2_cpp/typesupport_helpers.hpp>
#include <rosbag2_cpp/writer.hpp>
#include <rosbag2_cpp/writers/sequential_writer.hpp>
#include <rosbag2_storage/logging.hpp>
#include <rosbag2_storage/serialized_bag_message.hpp>
#include <rosbag2_storage/storage_options.hpp>
#include <tf2/LinearMath/Quaternion.h> // to Quaternion_to_euler
#include <tf2/LinearMath/Matrix3x3.h>  // to Quaternion_to_euler
#include <tf2/transform_datatypes.h>   // createQuaternionFromRPY
#include <tf2_eigen/tf2_eigen.hpp>  // tfs <-> eigen
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/transform_listener.h>
#include <std_msgs/msg/string.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
///// GTSAM
#include <gtsam/geometry/Rot3.h>
#include <gtsam/geometry/Point3.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/nonlinear/ISAM2.h>
///// coded headers
#include "loop_closure.h"
#include "pose_pcd.hpp"
#include "utilities.hpp"

#ifdef FASTLIO_SAM_QN_PARAM_DEBUG
#define GET_PARAM_DEBUG(name, param)                                           \
  this->get_parameter(name, param);                                            \
  RCLCPP_INFO_STREAM(this->get_logger(), name << ": " << param);
#else
#define GET_PARAM_DEBUG(name, param) this->get_parameter(name, param);
#endif

namespace fs = std::filesystem;
using namespace std::chrono;
typedef message_filters::sync_policies::ApproximateTime<nav_msgs::msg::Odometry, sensor_msgs::msg::PointCloud2> odom_pcd_sync_pol;

////////////////////////////////////////////////////////////////////////////////////////////////////
class FastLioSamScQn : public rclcpp::Node
{

public:
    using PointCloudT = sensor_msgs::msg::PointCloud2;
    using PathT = nav_msgs::msg::Path;
    using MarkerT = visualization_msgs::msg::Marker;
    using PoseStampedT = geometry_msgs::msg::PoseStamped;
    using OdomT = nav_msgs::msg::Odometry;
    using StringT = std_msgs::msg::String;

private:
    ///// basic params
    std::string map_frame_;
    std::string package_path_;
    std::string seq_name_;
    ///// shared data - odom and pcd
    std::mutex realtime_pose_mutex_, keyframes_mutex_;
    std::mutex graph_mutex_, vis_mutex_;
    Eigen::Matrix4d last_corrected_pose_ = Eigen::Matrix4d::Identity();
    Eigen::Matrix4d odom_delta_ = Eigen::Matrix4d::Identity();
    PosePcd current_frame_;
    std::vector<PosePcd> keyframes_;
    int current_keyframe_idx_ = 0;
    ///// graph and values
    bool is_initialized_ = false;
    bool loop_added_flag_ = false;     // for opt
    bool loop_added_flag_vis_ = false; // for vis
    std::shared_ptr<gtsam::ISAM2> isam_handler_ = nullptr;
    gtsam::NonlinearFactorGraph gtsam_graph_;
    gtsam::Values init_esti_;
    gtsam::Values corrected_esti_;
    double keyframe_thr_;
    double voxel_res_;
    int sub_key_num_;
    std::vector<std::pair<size_t, size_t>> loop_idx_pairs_; // for vis
    ///// visualize
    tf2_ros::Buffer tfListener_buffer_;
    tf2_ros::TransformBroadcaster broadcaster_;
    tf2_ros::TransformListener tfListener_;
    pcl::PointCloud<pcl::PointXYZ> odoms_, corrected_odoms_;
    nav_msgs::msg::Path odom_path_, corrected_path_;
    bool global_map_vis_switch_ = true;
    ///// results
    bool save_map_bag_ = false, save_map_pcd_ = false, save_in_kitti_format_ = false;
    std::string save_map_path_ = ROOT_DIR;
    ///// ros
    rclcpp::Publisher<PointCloudT>::SharedPtr odom_pub_;
    rclcpp::Publisher<PathT>::SharedPtr path_pub_;
    rclcpp::Publisher<PointCloudT>::SharedPtr corrected_odom_pub_;
    rclcpp::Publisher<PathT>::SharedPtr corrected_path_pub_;
    rclcpp::Publisher<PointCloudT>::SharedPtr corrected_pcd_map_pub_;
    rclcpp::Publisher<PointCloudT>::SharedPtr corrected_current_pcd_pub_;
    rclcpp::Publisher<MarkerT>::SharedPtr loop_detection_pub_;
    rclcpp::Publisher<PoseStampedT>::SharedPtr realtime_pose_pub_;
    rclcpp::Publisher<PointCloudT>::SharedPtr debug_src_pub_;
    rclcpp::Publisher<PointCloudT>::SharedPtr debug_dst_pub_;
    rclcpp::Publisher<PointCloudT>::SharedPtr debug_coarse_aligned_pub_;
    rclcpp::Publisher<PointCloudT>::SharedPtr debug_fine_aligned_pub_;

    rclcpp::Subscription<StringT>::SharedPtr sub_save_flag_;

    rclcpp::TimerBase::SharedPtr loop_timer_, vis_timer_;
    // odom, pcd sync, and save flag subscribers
    std::shared_ptr<message_filters::Synchronizer<odom_pcd_sync_pol>>
        sub_odom_pcd_sync_ = nullptr;
    std::shared_ptr<message_filters::Subscriber<nav_msgs::msg::Odometry>>
        sub_odom_ = nullptr;
    std::shared_ptr<message_filters::Subscriber<sensor_msgs::msg::PointCloud2>>
        sub_pcd_ = nullptr;
    ///// Loop closure
    std::shared_ptr<LoopClosure> loop_closure_;

public:
    explicit FastLioSamScQn();
    ~FastLioSamScQn();

private:
    // methods
    void updateOdomsAndPaths(const PosePcd &pose_pcd_in);
    bool checkIfKeyframe(const PosePcd &pose_pcd_in, const PosePcd &latest_pose_pcd);
    visualization_msgs::msg::Marker getLoopMarkers(const gtsam::Values &corrected_esti_in);
    // cb
    void odomPcdCallback(const nav_msgs::msg::Odometry::ConstSharedPtr &odom_msg,
                         const sensor_msgs::msg::PointCloud2::ConstSharedPtr &pcd_msg);
    void saveFlagCallback(const std_msgs::msg::String::ConstSharedPtr &msg);
    void loopTimerFunc();
    void visTimerFunc();
};


#endif
