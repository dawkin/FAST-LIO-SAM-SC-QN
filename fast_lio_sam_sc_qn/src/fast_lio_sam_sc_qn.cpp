#include "fast_lio_sam_sc_qn.h"
#include <ament_index_cpp/get_package_share_directory.hpp>
#include <ament_index_cpp/get_package_prefix.hpp>

using rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface;

FastLioSamScQn::FastLioSamScQn(): rclcpp_lifecycle::LifecycleNode("FastLioSamScQn")
{
    ////// ROS params
    LoopClosureConfig lc_config;
    auto &gc = lc_config.gicp_config_;
    auto &qc = lc_config.quatro_config_;
    /* basic */
    this->declare_parameter("basic.map_frame", "map");
    this->declare_parameter("basic.robot_frame", "robot");
    this->declare_parameter("basic.loop_update_hz", 2.0);
    this->declare_parameter("basic.vis_hz", 1.0);
    this->declare_parameter("basic.use_gravity_alignment", use_gravity_alignment_);
    this->declare_parameter("save_voxel_resolution", voxel_res_);
    this->declare_parameter("quatro_nano_gicp_voxel_resolution",lc_config.voxel_res_);
    /* keyframe */
    this->declare_parameter("keyframe.keyframe_threshold", keyframe_thr_);
    this->declare_parameter("keyframe.nusubmap_keyframes", lc_config.num_submap_keyframes_);
    this->declare_parameter("keyframe.enable_submap_matching", lc_config.enable_submap_matching_);
    /* ScanContext */
    this->declare_parameter("scancontext_max_correspondence_distance", lc_config.scancontext_max_correspondence_distance_);
    /* nano (GICP config) */
    this->declare_parameter("nano_gicp.thread_number", gc.nano_thread_number_);
    this->declare_parameter("nano_gicp.icp_score_threshold", gc.icp_score_thr_);
    this->declare_parameter("nano_gicp.correspondences_number",
                            gc.nano_correspondences_number_);
    this->declare_parameter("nano_gicp.max_iter", gc.nano_max_iter_);
    this->declare_parameter("nano_gicp.transformation_epsilon",
                            gc.transformation_epsilon_);
    this->declare_parameter("nano_gicp.euclidean_fitness_epsilon",
                            gc.euclidean_fitness_epsilon_);
    this->declare_parameter("nano_gicp.ransac.max_iter",
                            gc.nano_ransac_max_iter_);
    this->declare_parameter("nano_gicp.ransac.outlier_rejection_threshold",
                            gc.ransac_outlier_rejection_threshold_);
    /* quatro (Quatro config) */
    this->declare_parameter("quatro.enable", lc_config.enable_quatro_);
    this->declare_parameter("quatro.optimize_matching",
                            qc.use_optimized_matching_);
    this->declare_parameter("quatro.distance_threshold",
                            qc.quatro_distance_threshold_);
    this->declare_parameter("quatro.max_nucorrespondences",
                            qc.quatro_max_num_corres_);
    this->declare_parameter("quatro.fpfh_normal_radius",
                            qc.fpfh_normal_radius_);
    this->declare_parameter("quatro.fpfh_radius", qc.fpfh_radius_);
    this->declare_parameter("quatro.estimating_scale", qc.estimat_scale_);
    this->declare_parameter("quatro.noise_bound", qc.noise_bound_);
    this->declare_parameter("quatro.rotation.gnc_factor", qc.rot_gnc_factor_);
    this->declare_parameter("quatro.rotation.rot_cost_diff_threshold",
                            qc.rot_cost_diff_thr_);
    this->declare_parameter("quatro.rotation.numax_iter", qc.quatro_max_iter_);
    /* results */
    this->declare_parameter("result.save_map_bag", save_map_bag_);
    this->declare_parameter("result.save_map_pcd", save_map_pcd_);
    this->declare_parameter("result.save_map_path", ROOT_DIR);
    this->declare_parameter("result.save_in_kitti_format",
                            save_in_kitti_format_);
    this->declare_parameter("result.seq_name", seq_name_);
    /* offline */
    this->declare_parameter("offline.bag_file", "");
    this->declare_parameter("offline.fast_lio_config", "mid360.yaml");
    this->declare_parameter("offline.post_loop_optimization", false);
    this->declare_parameter("offline.buffer_time_sec", 2.0);

    RCLCPP_INFO(this->get_logger(), "ctor, parameters declared");
}

void FastLioSamScQn::initPublishersAndSubscribers()
{
  broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(
    this->shared_from_this());

  static_tf_broadcaster_ = std::make_unique<tf2_ros::StaticTransformBroadcaster>(
    this->shared_from_this());

  // Publishers
  corrected_pcd_map_pub_     = this->create_publisher<PointCloudT>("/corrected_map", 10);
  corrected_current_pcd_pub_ = this->create_publisher<PointCloudT>("/corrected_current_pcd", 10);
  loop_detection_pub_        = this->create_publisher<MarkerT>("/loop_detection", 10);
  realtime_pose_pub_         = this->create_publisher<PoseStampedT>("/pose_stamped", 10);
  debug_src_pub_             = this->create_publisher<PointCloudT>("/src", 10);
  debug_dst_pub_             = this->create_publisher<PointCloudT>("/dst", 10);
  debug_coarse_aligned_pub_  = this->create_publisher<PointCloudT>("/coarse_aligned_quatro", 10);
  debug_fine_aligned_pub_    = this->create_publisher<PointCloudT>("/fine_aligned_nano_gicp", 10);
  odom_pub_              = this->create_publisher<PointCloudT>("/ori_odom", 10);
  path_pub_              = this->create_publisher<nav_msgs::msg::Path>("/ori_path", 10);
  corrected_odom_pub_    = this->create_publisher<PointCloudT>("/corrected_odom", 10);
  corrected_path_pub_    = this->create_publisher<nav_msgs::msg::Path>("/corrected_path", 10);

  clock_pub_ = this->create_publisher<rosgraph_msgs::msg::Clock>(
      "/clock",
      rclcpp::QoS(rclcpp::KeepLast(10)).best_effort().durability_volatile());

  pub_ext_lat_ = this->create_publisher<std_msgs::msg::Float64>("/ext/lat", 10);
  pub_ext_lon_ = this->create_publisher<std_msgs::msg::Float64>("/ext/lon", 10);
}

LifecycleNodeInterface::CallbackReturn FastLioSamScQn::on_configure(const rclcpp_lifecycle::State&)
{
    RCLCPP_INFO(this->get_logger(), "on_configure()");

    double loop_update_hz, vis_hz;
    LoopClosureConfig lc_config;
    auto &gc = lc_config.gicp_config_;
    auto &qc = lc_config.quatro_config_;

    /* basic */
    this->get_parameter("basic.map_frame", map_frame_);
    this->get_parameter("basic.robot_frame", robot_frame_);
    this->get_parameter("basic.loop_update_hz", loop_update_hz);
    this->get_parameter("basic.vis_hz", vis_hz);
    this->get_parameter("basic.use_gravity_alignment", use_gravity_alignment_);
    this->get_parameter("save_voxel_resolution", voxel_res_);
    this->get_parameter("quatro_nano_gicp_voxel_resolution", lc_config.voxel_res_);
    /* keyframe */
    this->get_parameter("keyframe.keyframe_threshold", keyframe_thr_);
    this->get_parameter("keyframe.nusubmap_keyframes",
                    lc_config.num_submap_keyframes_);
    this->get_parameter("keyframe.enable_submap_matching",
                    lc_config.enable_submap_matching_);

    /* ScanContext */
    this->get_parameter("scancontext_max_correspondence_distance", lc_config.scancontext_max_correspondence_distance_);

    /* nano (GICP config) */
    this->get_parameter("nano_gicp.thread_number", gc.nano_thread_number_);
    this->get_parameter("nano_gicp.icp_score_threshold", gc.icp_score_thr_);
    this->get_parameter("nano_gicp.correspondences_number",
                    gc.nano_correspondences_number_);
    this->get_parameter("nano_gicp.max_iter", gc.nano_max_iter_);
    this->get_parameter("nano_gicp.transformation_epsilon",
                    gc.transformation_epsilon_);
    this->get_parameter("nano_gicp.euclidean_fitness_epsilon",
                    gc.euclidean_fitness_epsilon_);
    this->get_parameter("nano_gicp.ransac.max_iter", gc.nano_ransac_max_iter_);
    this->get_parameter("nano_gicp.ransac.outlier_rejection_threshold",
                    gc.ransac_outlier_rejection_threshold_);
    /* quatro (Quatro config) */
    this->get_parameter("quatro.enable", lc_config.enable_quatro_);
    this->get_parameter("quatro.optimize_matching", qc.use_optimized_matching_);
    this->get_parameter("quatro.distance_threshold", qc.quatro_distance_threshold_);
    this->get_parameter("quatro.max_nucorrespondences", qc.quatro_max_num_corres_);
    this->get_parameter("quatro.fpfh_normal_radius", qc.fpfh_normal_radius_);
    this->get_parameter("quatro.fpfh_radius", qc.fpfh_radius_);
    this->get_parameter("quatro.estimating_scale", qc.estimat_scale_);
    this->get_parameter("quatro.noise_bound", qc.noise_bound_);
    this->get_parameter("quatro.rotation.gnc_factor", qc.rot_gnc_factor_);
    this->get_parameter("quatro.rotation.rot_cost_diff_threshold",
                    qc.rot_cost_diff_thr_);
    this->get_parameter("quatro.rotation.numax_iter", qc.quatro_max_iter_);
    /* results */
    this->get_parameter("result.save_map_bag", save_map_bag_);
    this->get_parameter("result.save_map_pcd", save_map_pcd_);
    this->get_parameter("result.save_map_path", save_map_path_);
    this->get_parameter("result.save_in_kitti_format", save_in_kitti_format_);
    this->get_parameter("result.seq_name", seq_name_);
    /* Offline */
    this->get_parameter("offline.bag_file", bag_file_);
    this->get_parameter("offline.fast_lio_config", fast_lio_config_);
    this->get_parameter("offline.post_loop_optimization", offline_post_loop_optimization_);
    this->get_parameter("offline.buffer_time_sec", bag_buffer_time_sec_);

    // Load the fast_lio config to instantiate fast_lio_core
    std::string fast_lio_pkg_path;
    try {
        fast_lio_pkg_path = ament_index_cpp::get_package_share_directory("fast_lio");
    } catch (const ament_index_cpp::PackageNotFoundError& e) {
        RCLCPP_ERROR(this->get_logger(), "fast_lio package not found: %s", e.what());
        return LifecycleNodeInterface::CallbackReturn::FAILURE;
    }

    std::string config_file_path = fast_lio_pkg_path + "/config/" + fast_lio_config_;
    YAML::Node config_yaml;
    try {
        config_yaml = YAML::LoadFile(config_file_path);
        RCLCPP_INFO(this->get_logger(), "Loaded fast_lio config: %s", config_file_path.c_str());
    } catch (const YAML::BadFile & e) {
        RCLCPP_ERROR(this->get_logger(), "Failed to load fast_lio config file: %s", config_file_path.c_str());
        return LifecycleNodeInterface::CallbackReturn::FAILURE;
    }

    YAML::Node params = config_yaml.begin()->second["ros__parameters"];
    if (!params) {
        RCLCPP_ERROR(this->get_logger(), "Could not find 'ros__parameters' in %s", config_file_path.c_str());
        return LifecycleNodeInterface::CallbackReturn::FAILURE;
    }

    try {
        fast_lio_config_struct_.point_filter_num = params["point_filter_num"].as<int>();
        fast_lio_config_struct_.max_iteration = params["max_iteration"].as<int>();
        fast_lio_config_struct_.filter_size_surf = params["filter_size_surf"].as<double>();
        fast_lio_config_struct_.filter_size_map = params["filter_size_map"].as<double>();
        fast_lio_config_struct_.cube_side_length = params["cube_side_length"].as<double>();
        fast_lio_config_struct_.runtime_pos_log_enable = params["runtime_pos_log_enable"].as<bool>();
        YAML::Node common_params = params["common"];
        fast_lio_config_struct_.time_sync_en = common_params["time_sync_en"].as<bool>();
        fast_lio_config_struct_.time_offset_lidar_to_imu = common_params["time_offset_lidar_to_imu"].as<double>();
        // Get topic names for subscribers
        fast_lio_lidar_topic_ = common_params["lid_topic"].as<std::string>();
        fast_lio_imu_topic_ = common_params["imu_topic"].as<std::string>();

        YAML::Node preprocess_params = params["preprocess"];
        fast_lio_config_struct_.lidar_type = preprocess_params["lidar_type"].as<int>();
        fast_lio_config_struct_.scan_line = preprocess_params["scan_line"].as<int>();
        fast_lio_config_struct_.blind = preprocess_params["blind"].as<double>();
        fast_lio_config_struct_.timestamp_unit = preprocess_params["timestamp_unit"].as<int>();
        fast_lio_config_struct_.scan_rate = preprocess_params["scan_rate"].as<int>();
        YAML::Node mapping_params = params["mapping"];
        fast_lio_config_struct_.acc_cov = mapping_params["acc_cov"].as<double>();
        fast_lio_config_struct_.gyr_cov = mapping_params["gyr_cov"].as<double>();
        fast_lio_config_struct_.b_acc_cov = mapping_params["b_acc_cov"].as<double>();
        fast_lio_config_struct_.b_gyr_cov = mapping_params["b_gyr_cov"].as<double>();
        fast_lio_config_struct_.fov_degree = mapping_params["fov_degree"].as<double>();
        fast_lio_config_struct_.det_range = mapping_params["det_range"].as<double>();
        fast_lio_config_struct_.extrinsic_est_en = mapping_params["extrinsic_est_en"].as<bool>();
        fast_lio_config_struct_.extrinsic_T = mapping_params["extrinsic_T"].as<std::vector<double>>();
        fast_lio_config_struct_.extrinsic_R = mapping_params["extrinsic_R"].as<std::vector<double>>();
        fast_lio_config_struct_.pcd_save_en = false; // Disable fast_lio's own pcd saving
        fast_lio_config_struct_.log_path = "/tmp/";
        fast_lio_config_struct_.dense_publish_en = false;
        fast_lio_config_struct_.map_pub_en = true;
    } catch (const YAML::Exception &e) {
        RCLCPP_ERROR(this->get_logger(), "Error while parsing fast_lio YAML file: %s", e.what());
        return LifecycleNodeInterface::CallbackReturn::FAILURE;
    }

    fast_lio_core_ = std::make_unique<FastLioCore>(fast_lio_config_struct_);
    RCLCPP_INFO(this->get_logger(), "FastLioCore instantiated.");

    loop_closure_.reset(new LoopClosure(lc_config));

    initPublishersAndSubscribers();

    /* Initialization of GTSAM */
    gtsam::ISAM2Params isam_params_;
    isam_params_.relinearizeThreshold = 0.01;
    isam_params_.relinearizeSkip = 1;
    isam_handler_ = std::make_shared<gtsam::ISAM2>(isam_params_);
    /* ROS things */
    odom_path_.header.frame_id = map_frame_;
    corrected_path_.header.frame_id = map_frame_;
    package_path_ = ament_index_cpp::get_package_share_directory("fast_lio_sam_sc_qn");

    RCLCPP_INFO(this->get_logger(), "Configured");

    return rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn::SUCCESS;
}

LifecycleNodeInterface::CallbackReturn FastLioSamScQn::on_activate(const rclcpp_lifecycle::State&)
{
    RCLCPP_INFO(this->get_logger(), "on_activate()");

    if (!bag_file_.empty())
    {
        // Offline mode
        RCLCPP_INFO(this->get_logger(), "Bag file provided [%s], running in Offline mode.", bag_file_.c_str());
        std::thread offline_thread(&FastLioSamScQn::runOffline, this);
        offline_thread.detach();
    }
    else
    {
        // Online mode
        RCLCPP_INFO(this->get_logger(), "No bag file provided, running in Online mode.");

        // Subscribers for fast_lio_core
        RCLCPP_INFO(this->get_logger(), "Subscribing to IMU topic: %s", fast_lio_imu_topic_.c_str());
        sub_imu_ = this->create_subscription<sensor_msgs::msg::Imu>(
                fast_lio_imu_topic_, rclcpp::QoS(10).best_effort(),
                std::bind(&FastLioSamScQn::imuCallback, this, std::placeholders::_1));

        RCLCPP_INFO(this->get_logger(), "Subscribing to LiDAR topic: %s", fast_lio_lidar_topic_.c_str());
        if (fast_lio_config_struct_.lidar_type == AVIA)
        {
            RCLCPP_INFO(this->get_logger(), "Using Livox (AVIA) subscriber.");
            sub_lidar_livox_ = this->create_subscription<livox_ros_driver2::msg::CustomMsg>(
                    fast_lio_lidar_topic_, rclcpp::QoS(10).best_effort(),
                    std::bind(&FastLioSamScQn::lidarLivoxCallback, this, std::placeholders::_1));
        }
        else
        {
            RCLCPP_INFO(this->get_logger(), "Using standard PointCloud2 subscriber.");
            sub_lidar_pc2_ = this->create_subscription<PointCloudT>(
                    fast_lio_lidar_topic_, rclcpp::QoS(10).best_effort(),
                    std::bind(&FastLioSamScQn::lidarPc2Callback, this, std::placeholders::_1));
        }

        RCLCPP_INFO(this->get_logger(), "Waiting for IMU initialization...");
        /* Timers */
        loop_timer_ = this->create_wall_timer(500ms, std::bind(&FastLioSamScQn::loopTimerFunc, this));
        vis_timer_ = this->create_wall_timer(500ms, std::bind(&FastLioSamScQn::visTimerFunc, this));
    }

    return rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn::SUCCESS;
}

LifecycleNodeInterface::CallbackReturn FastLioSamScQn::on_deactivate(const rclcpp_lifecycle::State&)
{
    RCLCPP_INFO(this->get_logger(), "on_deactivate()");

    loop_timer_.reset();
    vis_timer_.reset();

    sub_imu_.reset();
    sub_lidar_pc2_.reset();
    sub_lidar_livox_.reset();

    saveMaps();

    return rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn::SUCCESS;
}

LifecycleNodeInterface::CallbackReturn FastLioSamScQn::on_cleanup(const rclcpp_lifecycle::State&)
{
    RCLCPP_INFO(this->get_logger(), "on_cleanup()");

    loop_timer_.reset();
    vis_timer_.reset();

    // Reset fast_lio subscribers
    sub_imu_.reset();
    sub_lidar_pc2_.reset();
    sub_lidar_livox_.reset();

    loop_closure_.reset();
    fast_lio_core_.reset();

    return rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn::SUCCESS;
}

LifecycleNodeInterface::CallbackReturn FastLioSamScQn::on_shutdown(const rclcpp_lifecycle::State&)
{
    RCLCPP_INFO(this->get_logger(), "on_shutdown()");
    rclcpp::shutdown();
    return rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn::SUCCESS;
}

void FastLioSamScQn::saveMaps()
{
    if (keyframes_.empty())
    {
        RCLCPP_WARN(this->get_logger(), "No keyframes to save. Exiting.");
        return;
    }

    if (save_map_bag_)
    {
        auto writer = std::make_unique<rosbag2_cpp::Writer>();
        try {
            writer->open(save_map_path_ + "map_bag");
        } catch (const std::exception& e) {
            RCLCPP_ERROR(this->get_logger(), "Failed to open bag file for writing: %s", e.what());
            return;
        }

        const std::string pose_topic_name = "/keyframe_pose";
        rosbag2_storage::TopicMetadata pose_topic_metadata;
        pose_topic_metadata.name = pose_topic_name;
        pose_topic_metadata.type = "geometry_msgs/msg/PoseStamped";
        pose_topic_metadata.serialization_format = rmw_get_serialization_format();
        writer->create_topic(pose_topic_metadata);

        const std::string pcd_topic_name = "/keyframe_pcd";
        rosbag2_storage::TopicMetadata pcd_topic_metadata;
        pcd_topic_metadata.name = pcd_topic_name;
        pcd_topic_metadata.type = "sensor_msgs/msg/PointCloud2";
        pcd_topic_metadata.serialization_format = rmw_get_serialization_format();
        writer->create_topic(pcd_topic_metadata);

        {
            std::lock_guard<std::mutex> lock(keyframes_mutex_);
            for (const auto& keyframe : keyframes_) {
                rclcpp::Time time = fromSec(keyframe.timestamp_);

                auto pose_msg = std::make_shared<geometry_msgs::msg::PoseStamped>(
                        poseEigToPoseStamped(keyframe.pose_corrected_eig_, map_frame_)
                        );
                pose_msg->header.stamp = time;
                writer->write(*pose_msg, pose_topic_name, time);

                auto pcd_msg = std::make_shared<sensor_msgs::msg::PointCloud2>(
                        pclToPclRos(keyframe.pcd_, map_frame_)
                        );
                pcd_msg->header.stamp = time;
                writer->write(*pcd_msg, pcd_topic_name, time);
            }
        }

        writer->close();
        RCLCPP_INFO(this->get_logger(), "\033[36;1mResult saved in .bag format!!!\033[0m");
    }

    if (save_map_pcd_)
    {
        pcl::PointCloud<PointType>::Ptr corrected_map(new pcl::PointCloud<PointType>());
        corrected_map->reserve(keyframes_[0].pcd_.size() * keyframes_.size()); // it's an approximated size
        {
            std::lock_guard<std::mutex> lock(keyframes_mutex_);
            for (size_t i = 0; i < keyframes_.size(); ++i)
            {
                *corrected_map += transformPcd(keyframes_[i].pcd_, keyframes_[i].pose_corrected_eig_);
            }
        }
        const auto &voxelized_map = voxelizePcd(corrected_map, voxel_res_);
        pcl::io::savePCDFileASCII<PointType>(save_map_path_ + "map.pcd", *voxelized_map);
        RCLCPP_INFO(this->get_logger(), "\033[32;1mResult saved in .pcd format!!!\033[0m");
    }
}

void FastLioSamScQn::performLoopClosureForKf(size_t keyframe_idx)
{
    auto& keyframe = keyframes_[keyframe_idx];
    if (keyframe.processed_) return;
    keyframe.processed_ = true;

    const int closest_keyframe_idx = loop_closure_->fetchCandidateKeyframeIdx(keyframe, keyframes_);
    if (closest_keyframe_idx < 0)
    {
        return;
    }

    const RegistrationOutput& reg_output = loop_closure_->performLoopClosure(keyframe, keyframes_, closest_keyframe_idx);
    if (reg_output.is_valid_)
    {
        RCLCPP_INFO(this->get_logger(), "\033[1;32mLoop closure found between kf %zu and %d. Score: %.3f\033[0m", keyframe_idx, closest_keyframe_idx, reg_output.score_);
        const auto& score = reg_output.score_;
        gtsam::Pose3 pose_from = poseEigToGtsamPose(reg_output.pose_between_eig_ * keyframe.pose_corrected_eig_);
        gtsam::Pose3 pose_to = poseEigToGtsamPose(keyframes_[closest_keyframe_idx].pose_corrected_eig_);
        auto variance_vector = (gtsam::Vector(6) << score, score, score, score, score, score).finished();
        gtsam::noiseModel::Diagonal::shared_ptr loop_noise = gtsam::noiseModel::Diagonal::Variances(variance_vector);
        {
            std::lock_guard<std::mutex> lock(graph_mutex_);
            gtsam_graph_.add(gtsam::BetweenFactor<gtsam::Pose3>(keyframe.idx_,
                                                                closest_keyframe_idx,
                                                                pose_from.between(pose_to),
                                                                loop_noise));
        }
        loop_idx_pairs_.push_back({keyframe.idx_, closest_keyframe_idx});
        loop_added_flag_ = true; // Signal that a loop has been added
    }
    else
    {
        RCLCPP_INFO(this->get_logger(), "\033[1;31mLoop closure between kf %zu and %d rejected. Score: %.3f\033[0m", keyframe_idx, closest_keyframe_idx, reg_output.score_);
    }
}

void FastLioSamScQn::runOffline()
{
    RCLCPP_INFO(this->get_logger(), "Starting offline processing from bag: %s", bag_file_.c_str());
    rclcpp::Rate rate(200.0);

    rosbag2_storage::StorageOptions storage_options({bag_file_, ""});
    rosbag2_cpp::ConverterOptions converter_options;
    rosbag2_cpp::readers::SequentialReader reader;
    try {
        reader.open(storage_options, converter_options);
    } catch (const std::exception& e) {
        RCLCPP_ERROR(this->get_logger(), "Failed to open bag file: %s", e.what());
        return;
    }

    // Use topic names from loaded config
    std::string lid_topic = fast_lio_lidar_topic_;
    std::string imu_topic = fast_lio_imu_topic_;
    std::string tf_topic = "/tf";
    std::string tf_static_topic = "/tf_static";
    std::string lat_topic = "/ext/lat";
    std::string lon_topic = "/ext/lon";

    rosbag2_storage::StorageFilter filter;
    filter.topics = {lid_topic, imu_topic, tf_topic, tf_static_topic, lat_topic, lon_topic};
    reader.set_filter(filter);

    rclcpp::Serialization<sensor_msgs::msg::Imu> imu_serialization;
    rclcpp::Serialization<livox_ros_driver2::msg::CustomMsg> livox_serialization;
    rclcpp::Serialization<sensor_msgs::msg::PointCloud2> pc2_serialization;
    rclcpp::Serialization<tf2_msgs::msg::TFMessage> tf_serialization;
    rclcpp::Serialization<std_msgs::msg::Float64> double_serialization;

    RCLCPP_INFO(this->get_logger(), "Offline mode with BUFFERED reading.");

    RCLCPP_INFO(this->get_logger(), "Reading initial messages for IMU initialization...");
    deque<sensor_msgs::msg::Imu::ConstSharedPtr> init_imu_data;
    while (reader.has_next() && init_imu_data.size() < INIT_IMU_COUNT) {
        auto serialized_msg = reader.read_next();
        if (serialized_msg->topic_name == imu_topic) {
            auto msg = std::make_shared<sensor_msgs::msg::Imu>();
            rclcpp::SerializedMessage extracted_serialized_msg(*serialized_msg->serialized_data);
            imu_serialization.deserialize_message(&extracted_serialized_msg, msg.get());
            init_imu_data.push_back(msg);
        }
    }

    if (init_imu_data.size() < INIT_IMU_COUNT) {
        RCLCPP_ERROR(this->get_logger(), "Not enough IMU messages in bag to initialize. Found %zu, need %d.", init_imu_data.size(), INIT_IMU_COUNT);
        return;
    }

    {
        std::lock_guard<std::mutex> lock(fast_lio_core_->mtx_buffer_);
        fast_lio_core_->imu_buffer_ = init_imu_data;
    }
    fast_lio_core_->initial_setup();
    RCLCPP_INFO(this->get_logger(), "IMU Initialized.");

    reader.seek(0);
    std::deque<StampedMessage> message_buffer;

    struct AuxMessage {
        double timestamp;
        std_msgs::msg::Float64 msg;
        std::string topic;
        bool operator<(const AuxMessage& other) const {
            return timestamp < other.timestamp;
        }
    };
    std::deque<AuxMessage> aux_buffer;

    while(reader.has_next() && rclcpp::ok()) {
        while(reader.has_next()) {
            if (!message_buffer.empty() && 
                    (message_buffer.back().timestamp - message_buffer.front().timestamp > bag_buffer_time_sec_)) {
                break; 
            }
            auto serialized_msg = reader.read_next();
            rclcpp::SerializedMessage extracted_serialized_msg(*serialized_msg->serialized_data);

            if (serialized_msg->topic_name == imu_topic) {
                auto msg = std::make_shared<sensor_msgs::msg::Imu>();
                imu_serialization.deserialize_message(&extracted_serialized_msg, msg.get());
                message_buffer.push_back({get_time_sec(msg->header.stamp), msg, nullptr, nullptr, imu_topic});
            } else if (serialized_msg->topic_name == lid_topic) {
                PointCloudXYZI::Ptr cloud(new PointCloudXYZI());
                double header_stamp = 0.0;
                if (fast_lio_config_struct_.lidar_type == AVIA) {
                    auto livox_msg = std::make_unique<livox_ros_driver2::msg::CustomMsg>();
                    livox_serialization.deserialize_message(&extracted_serialized_msg, livox_msg.get());
                    header_stamp = get_time_sec(livox_msg->header.stamp);
                    fast_lio_core_->p_pre_->process(std::move(livox_msg), cloud);
                } else {
                    auto pc2_msg = std::make_unique<sensor_msgs::msg::PointCloud2>();
                    pc2_serialization.deserialize_message(&extracted_serialized_msg, pc2_msg.get());
                    header_stamp = get_time_sec(pc2_msg->header.stamp);
                    fast_lio_core_->p_pre_->process(std::move(pc2_msg), cloud);
                }
                message_buffer.push_back({header_stamp, nullptr, cloud, nullptr, lid_topic});
            } else if (serialized_msg->topic_name == tf_topic || serialized_msg->topic_name == tf_static_topic) {
                auto msg = std::make_shared<tf2_msgs::msg::TFMessage>();
                tf_serialization.deserialize_message(&extracted_serialized_msg, msg.get());
                if (!msg->transforms.empty()) {
                    message_buffer.push_back({get_time_sec(msg->transforms[0].header.stamp), nullptr, nullptr, msg, serialized_msg->topic_name});
                }
            }
            else if (serialized_msg->topic_name == lat_topic || serialized_msg->topic_name == lon_topic) {
                std_msgs::msg::Float64 msg;
                double_serialization.deserialize_message(&extracted_serialized_msg, &msg);
                double t = (double)serialized_msg->recv_timestamp * 1e-9;
                aux_buffer.push_back({t, msg, serialized_msg->topic_name});
            }
        }

        std::stable_sort(message_buffer.begin(), message_buffer.end());
        std::stable_sort(aux_buffer.begin(), aux_buffer.end());

        double process_until_time = message_buffer.back().timestamp - (bag_buffer_time_sec_ / 2.0);
        if (!reader.has_next()) {
            process_until_time = std::numeric_limits<double>::max();
        }

        while (!message_buffer.empty() && message_buffer.front().timestamp < process_until_time) {
            StampedMessage stamped_msg = message_buffer.front();
            message_buffer.pop_front();

            // [Added] Publish aux messages up to the current simulation time
            while(!aux_buffer.empty() && aux_buffer.front().timestamp <= stamped_msg.timestamp) {
                if (aux_buffer.front().topic == lat_topic) {
                    pub_ext_lat_->publish(aux_buffer.front().msg);
                } else {
                    pub_ext_lon_->publish(aux_buffer.front().msg);
                }
                aux_buffer.pop_front();
            }

            rosgraph_msgs::msg::Clock clock_msg;
            clock_msg.clock = get_ros_time(stamped_msg.timestamp);
            clock_pub_->publish(clock_msg);

            { // Push to buffers
                std::lock_guard<std::mutex> lock(fast_lio_core_->mtx_buffer_);
                if (stamped_msg.topic_name == imu_topic) {
                    fast_lio_core_->imu_buffer_.push_back(stamped_msg.imu_msg);
                } else if (stamped_msg.topic_name == lid_topic) {
                    fast_lio_core_->lidar_buffer_.push_back(stamped_msg.lidar_msg);
                    fast_lio_core_->time_buffer_.push_back(stamped_msg.timestamp);
                } else if (stamped_msg.topic_name == tf_topic) {
                    broadcaster_->sendTransform(stamped_msg.tf_msg->transforms);
                } else if (stamped_msg.topic_name == tf_static_topic) {
                    static_tf_broadcaster_->sendTransform(stamped_msg.tf_msg->transforms);
                }
            }

            MeasureGroup meas;
            if (fast_lio_core_->sync_packages(meas)) {
                FrameResult result = fast_lio_core_->process_frame(meas);
                if (result.cloud.empty()) continue;

                nav_msgs::msg::Odometry odom_msg;
                geometry_msgs::msg::Quaternion quat;
                fast_lio_core_->get_publish_odometry(odom_msg, quat);
                odom_msg.header.stamp = get_ros_time(fast_lio_core_->get_lidar_end_time());
                odom_msg.header.frame_id = map_frame_;

                Eigen::Matrix4d pose_world = Eigen::Matrix4d::Identity();
                pose_world.block<3, 3>(0, 0) = Eigen::Quaterniond(odom_msg.pose.pose.orientation.w, odom_msg.pose.pose.orientation.x, odom_msg.pose.pose.orientation.y, odom_msg.pose.pose.orientation.z).toRotationMatrix();
                pose_world(0, 3) = odom_msg.pose.pose.position.x;
                pose_world(1, 3) = odom_msg.pose.pose.position.y;
                pose_world(2, 3) = odom_msg.pose.pose.position.z;

                pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_world(new pcl::PointCloud<pcl::PointXYZI>());
                pcl::transformPointCloud(result.cloud, *cloud_world, pose_world);

                sensor_msgs::msg::PointCloud2 pcd_msg;
                pcl::toROSMsg(*cloud_world, pcd_msg);
                pcd_msg.header.stamp = odom_msg.header.stamp;
                pcd_msg.header.frame_id = "body";

                odomPcdCallback(std::make_shared<nav_msgs::msg::Odometry>(odom_msg),
                        std::make_shared<sensor_msgs::msg::PointCloud2>(pcd_msg));

                nav_msgs::msg::Path live_corrected_path;
                live_corrected_path.header.frame_id = map_frame_;
                live_corrected_path.header.stamp = odom_msg.header.stamp;
                std::lock_guard<std::mutex> lock(keyframes_mutex_);
                for(const auto& kf : keyframes_) {
                    live_corrected_path.poses.push_back(
                            poseEigToPoseStamped(kf.pose_corrected_eig_, map_frame_)
                            );
                }
                corrected_path_pub_->publish(live_corrected_path);

                if (corrected_pcd_map_pub_->get_subscription_count() > 0 && !keyframes_.empty())
                {
                    pcl::PointCloud<PointType>::Ptr corrected_map(new pcl::PointCloud<PointType>());
                    corrected_map->reserve(keyframes_[0].pcd_.size() * keyframes_.size());

                    for (size_t i = 0; i < keyframes_.size(); ++i)
                    {
                        *corrected_map += transformPcd(keyframes_[i].pcd_, keyframes_[i].pose_corrected_eig_);
                    }

                    const auto &voxelized_map = voxelizePcd(corrected_map, voxel_res_);
                    corrected_pcd_map_pub_->publish(pclToPclRos(*voxelized_map, map_frame_));
                }
                rate.sleep();
            }
        }
    }

    // Final post loop optimization
    if (offline_post_loop_optimization_)
    {
        RCLCPP_INFO(this->get_logger(), "Finished processing all messages. Starting offline BATCH loop closure and optimization...");
        for (size_t i = 0; i < keyframes_.size(); ++i) {
            performLoopClosureForKf(i);
        }

        RCLCPP_INFO(this->get_logger(), "Performing final batch graph optimization...");
        {
            std::lock_guard<std::mutex> lock(graph_mutex_);
            isam_handler_->update(gtsam_graph_, init_esti_);
            for (int i = 0; i < 10; ++i) {
                isam_handler_->update();
            }
            gtsam_graph_.resize(0);
            init_esti_.clear();
        }
    }
    else
    {
        RCLCPP_INFO(this->get_logger(), "Finished processing all messages with INCREMENTAL loop closure.");
    }

    {
        std::lock_guard<std::mutex> lock(keyframes_mutex_);
        std::lock_guard<std::mutex> lock2(realtime_pose_mutex_);
        corrected_esti_ = isam_handler_->calculateEstimate();
        for (size_t i = 0; i < corrected_esti_.size(); ++i)
        {
            if (i < keyframes_.size()) {
                keyframes_[i].pose_corrected_eig_ = gtsamPoseToPoseEig(corrected_esti_.at<gtsam::Pose3>(i));
            }
        }
    }

    if (offline_post_loop_optimization_) {
        RCLCPP_INFO(this->get_logger(), "Publishing final optimized path...");
        nav_msgs::msg::Path final_path;
        final_path.header.frame_id = map_frame_;
        final_path.header.stamp = this->get_clock()->now();
        if (!keyframes_.empty()) {
            std::lock_guard<std::mutex> lock(keyframes_mutex_);
            final_path.header.stamp = rclcpp::Time(static_cast<int64_t>(keyframes_.back().timestamp_ * 1e9));
        }

        for (size_t i = 0; i < corrected_esti_.size(); ++i) {
            final_path.poses.push_back(
                gtsamPoseToPoseStamped(corrected_esti_.at<gtsam::Pose3>(i), map_frame_)
            );
        }
        corrected_path_pub_->publish(final_path);
    }

    visTimerFunc(); // final visualization call

    RCLCPP_INFO(this->get_logger(), "Offline processing finished. Final results are now available.");
    this->trigger_transition(lifecycle_msgs::msg::Transition::TRANSITION_DEACTIVATE);
    this->trigger_transition(lifecycle_msgs::msg::Transition::TRANSITION_INACTIVE_SHUTDOWN);
}

void FastLioSamScQn::imuCallback(const sensor_msgs::msg::Imu::ConstSharedPtr msg)
{
    if (!imu_initialized_)
    {
        init_imu_data_.push_back(msg);
        if (init_imu_data_.size() < INIT_IMU_COUNT)
        {
            RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
                "Waiting for IMU initialization... %zu / %d", init_imu_data_.size(), INIT_IMU_COUNT);
            return;
        }

        {
            std::lock_guard<std::mutex> lock(fast_lio_core_->mtx_buffer_);
            fast_lio_core_->imu_buffer_ = init_imu_data_;
        }
        fast_lio_core_->initial_setup();
        imu_initialized_ = true;
        RCLCPP_INFO(this->get_logger(), "IMU Initialized for Live Mode.");
        return; // Don't process this first message
    }

    // Push to buffer
    std::lock_guard<std::mutex> lock(fast_lio_core_->mtx_buffer_);
    fast_lio_core_->imu_buffer_.push_back(msg);
}

void FastLioSamScQn::lidarPc2Callback(const sensor_msgs::msg::PointCloud2::ConstSharedPtr msg)
{
    if (!imu_initialized_) return;

    auto pc2_msg_copy = std::make_unique<sensor_msgs::msg::PointCloud2>(*msg);

    PointCloudXYZI::Ptr cloud(new PointCloudXYZI());
    double header_stamp = get_time_sec(pc2_msg_copy->header.stamp);
    fast_lio_core_->p_pre_->process(std::move(pc2_msg_copy), cloud);
    processLidarData(header_stamp, cloud);
}

void FastLioSamScQn::lidarLivoxCallback(const livox_ros_driver2::msg::CustomMsg::ConstSharedPtr msg)
{
    if (!imu_initialized_) return;

    auto livox_msg_copy = std::make_unique<livox_ros_driver2::msg::CustomMsg>(*msg);

    PointCloudXYZI::Ptr cloud(new PointCloudXYZI());
    double header_stamp = get_time_sec(livox_msg_copy->header.stamp);
    fast_lio_core_->p_pre_->process(std::move(livox_msg_copy), cloud);
    processLidarData(header_stamp, cloud);
}

void FastLioSamScQn::processLidarData(double header_stamp, PointCloudXYZI::Ptr cloud)
{
    {
        std::lock_guard<std::mutex> lock(fast_lio_core_->mtx_buffer_);
        fast_lio_core_->lidar_buffer_.push_back(cloud);
        fast_lio_core_->time_buffer_.push_back(header_stamp);
    }

    MeasureGroup meas;
    if (fast_lio_core_->sync_packages(meas))
    {
        FrameResult result = fast_lio_core_->process_frame(meas);
        if (result.cloud.empty()) return;

        nav_msgs::msg::Odometry odom_msg;
        geometry_msgs::msg::Quaternion quat;
        fast_lio_core_->get_publish_odometry(odom_msg, quat);
        odom_msg.header.stamp = get_ros_time(fast_lio_core_->get_lidar_end_time());
        odom_msg.header.frame_id = map_frame_;

        Eigen::Matrix4d pose_world = Eigen::Matrix4d::Identity();
        pose_world.block<3, 3>(0, 0) = Eigen::Quaterniond(odom_msg.pose.pose.orientation.w, odom_msg.pose.pose.orientation.x, odom_msg.pose.pose.orientation.y, odom_msg.pose.pose.orientation.z).toRotationMatrix();
        pose_world(0, 3) = odom_msg.pose.pose.position.x;
        pose_world(1, 3) = odom_msg.pose.pose.position.y;
        pose_world(2, 3) = odom_msg.pose.pose.position.z;

        pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_world(new pcl::PointCloud<pcl::PointXYZI>());
        pcl::transformPointCloud(result.cloud, *cloud_world, pose_world);

        sensor_msgs::msg::PointCloud2 pcd_msg;
        pcl::toROSMsg(*cloud_world, pcd_msg);
        pcd_msg.header.stamp = odom_msg.header.stamp;
        pcd_msg.header.frame_id = "body";

        odomPcdCallback(std::make_shared<nav_msgs::msg::Odometry>(odom_msg),
                        std::make_shared<sensor_msgs::msg::PointCloud2>(pcd_msg));
    }
}

void FastLioSamScQn::odomPcdCallback(const nav_msgs::msg::Odometry::ConstSharedPtr &odom_msg,
                                     const sensor_msgs::msg::PointCloud2::ConstSharedPtr &pcd_msg)
{
    Eigen::Matrix4d last_odom_tf;
    last_odom_tf = current_frame_.pose_eig_;                              // to calculate delta
    current_frame_ = PosePcd(*odom_msg, *pcd_msg, current_keyframe_idx_); // to be checked if keyframe or not

    if (use_gravity_alignment_)
    {
        // Downward Z-axis
        Eigen::Vector3d z_down(0.0, 0.0, -1.0);

        // Retrieve current gravity estimate from the fast_lio state
        auto estimated_gravity = fast_lio_core_->kf_.get_x().grav;
        Eigen::Vector3d g_vec(estimated_gravity[0], estimated_gravity[1], estimated_gravity[2]);

        // Fast-LIO state.grav estimates the gravity vector in the local world frame
        // We rotate this vector to align with z_down, the standard downward Z axis
        Eigen::Matrix4d align_tf = Eigen::Matrix4d::Identity();
        align_tf.block<3,3>(0,0) = getRotationToAlignVectors(g_vec, z_down);

        // Apply alignment to current frame
        current_frame_.pose_eig_ = align_tf * current_frame_.pose_eig_;
        current_frame_.pose_corrected_eig_ = align_tf * current_frame_.pose_corrected_eig_;
    }

    high_resolution_clock::time_point t1 = high_resolution_clock::now();
    {
        //// 1. realtime pose = last corrected odom * delta (last -> current)
        std::lock_guard<std::mutex> lock(realtime_pose_mutex_);
        odom_delta_ = odom_delta_ * last_odom_tf.inverse() * current_frame_.pose_eig_;
        current_frame_.pose_corrected_eig_ = last_corrected_pose_ * odom_delta_;
        realtime_pose_pub_-> publish(poseEigToPoseStamped(current_frame_.pose_corrected_eig_, map_frame_));
        geometry_msgs::msg::TransformStamped trans_stamped_msg_;
        trans_stamped_msg_.transform =
            tf2::toMsg(poseEigToROSTf(current_frame_.pose_corrected_eig_));
        trans_stamped_msg_.header.frame_id = map_frame_;
        trans_stamped_msg_.child_frame_id = robot_frame_;
        trans_stamped_msg_.header.stamp = odom_msg->header.stamp;

        broadcaster_->sendTransform(trans_stamped_msg_);
    }
    corrected_current_pcd_pub_->publish(pclToPclRos(transformPcd(current_frame_.pcd_, current_frame_.pose_corrected_eig_), map_frame_));

    if (!is_initialized_) //// init only once
    {
        // others
        keyframes_.push_back(current_frame_);
        updateOdomsAndPaths(current_frame_);
        // graph
        auto variance_vector = (gtsam::Vector(6) << 1e-4, 1e-4, 1e-4, 1e-2, 1e-2, 1e-2).finished(); // rad*rad,
                                                                                                    // meter*meter
        gtsam::noiseModel::Diagonal::shared_ptr prior_noise = gtsam::noiseModel::Diagonal::Variances(variance_vector);
        gtsam_graph_.add(gtsam::PriorFactor<gtsam::Pose3>(0, poseEigToGtsamPose(current_frame_.pose_eig_), prior_noise));
        init_esti_.insert(current_keyframe_idx_, poseEigToGtsamPose(current_frame_.pose_eig_));
        current_keyframe_idx_++;
        // ScanContext
        loop_closure_->updateScancontext(current_frame_.pcd_);
        is_initialized_ = true;
    }
    else
    {
        //// 2. check if keyframe
        high_resolution_clock::time_point t2 = high_resolution_clock::now();
        if (checkIfKeyframe(current_frame_, keyframes_.back()))
        {
            // 2-2. if so, save
            {
                std::lock_guard<std::mutex> lock(keyframes_mutex_);
                keyframes_.push_back(current_frame_);
            }
            // 2-3. if so, add to graph
            auto variance_vector = (gtsam::Vector(6) << 1e-4, 1e-4, 1e-4, 1e-2, 1e-2, 1e-2).finished();
            gtsam::noiseModel::Diagonal::shared_ptr odom_noise = gtsam::noiseModel::Diagonal::Variances(variance_vector);
            gtsam::Pose3 pose_from = poseEigToGtsamPose(keyframes_[current_keyframe_idx_ - 1].pose_corrected_eig_);
            gtsam::Pose3 pose_to = poseEigToGtsamPose(current_frame_.pose_corrected_eig_);
            {
                std::lock_guard<std::mutex> lock(graph_mutex_);
                gtsam_graph_.add(gtsam::BetweenFactor<gtsam::Pose3>(current_keyframe_idx_ - 1,
                                                                    current_keyframe_idx_,
                                                                    pose_from.between(pose_to),
                                                                    odom_noise));
                init_esti_.insert(current_keyframe_idx_, pose_to);
            }

            if (!bag_file_.empty() && !offline_post_loop_optimization_)
            {
                performLoopClosureForKf(keyframes_.size() - 1);
            }

            current_keyframe_idx_++;
            // 2-4. if so, update ScanContext
            loop_closure_->updateScancontext(current_frame_.pcd_);

            //// 3. vis
            high_resolution_clock::time_point t3 = high_resolution_clock::now();
            {
                std::lock_guard<std::mutex> lock(vis_mutex_);
                updateOdomsAndPaths(current_frame_);
            }

            //// 4. optimize with graph
            high_resolution_clock::time_point t4 = high_resolution_clock::now();
            // m_corrected_esti = gtsam::LevenbergMarquardtOptimizer(m_gtsam_graph, init_esti_).optimize(); // cf. isam.update vs values.LM.optimize
            {
                std::lock_guard<std::mutex> lock(graph_mutex_);
                isam_handler_->update(gtsam_graph_, init_esti_);
                isam_handler_->update();
                if (loop_added_flag_) // https://github.com/TixiaoShan/LIO-SAM/issues/5#issuecomment-653752936
                {
                    isam_handler_->update();
                    isam_handler_->update();
                    isam_handler_->update();
                }
                gtsam_graph_.resize(0);
                init_esti_.clear();
            }

            //// 5. handle corrected results
            // get corrected poses and reset odom delta (for realtime pose pub)
            high_resolution_clock::time_point t5 = high_resolution_clock::now();
            {
                std::lock_guard<std::mutex> lock(realtime_pose_mutex_);
                corrected_esti_ = isam_handler_->calculateEstimate();
                last_corrected_pose_ = gtsamPoseToPoseEig(corrected_esti_.at<gtsam::Pose3>(corrected_esti_.size() - 1));
                odom_delta_ = Eigen::Matrix4d::Identity();
            }
            // correct poses in keyframes
            if (loop_added_flag_)
            {
                std::lock_guard<std::mutex> lock(keyframes_mutex_);
                for (size_t i = 0; i < corrected_esti_.size(); ++i)
                {
                    keyframes_[i].pose_corrected_eig_ = gtsamPoseToPoseEig(corrected_esti_.at<gtsam::Pose3>(i));
                }
                loop_added_flag_ = false;
            }
            high_resolution_clock::time_point t6 = high_resolution_clock::now();

            RCLCPP_INFO(
                this->get_logger(),
                "real: %.1f, key_add: %.1f, vis: %.1f, opt: %.1f, res: %.1f, "
                "tot: %.1fms",
                duration_cast<microseconds>(t2 - t1).count() / 1e3,
                duration_cast<microseconds>(t3 - t2).count() / 1e3,
                duration_cast<microseconds>(t4 - t3).count() / 1e3,
                duration_cast<microseconds>(t5 - t4).count() / 1e3,
                duration_cast<microseconds>(t6 - t5).count() / 1e3,
                duration_cast<microseconds>(t6 - t1).count() / 1e3);
        }
    }
    return;
}

void FastLioSamScQn::loopTimerFunc()
{
    if (keyframes_.empty() || !is_initialized_)
    {
        return;
    }

    performLoopClosureForKf(keyframes_.size() - 1);

    if (loop_closure_->getClosestKeyframeidx() >= 0) {
        debug_src_pub_->publish(pclToPclRos(loop_closure_->getSourceCloud(), map_frame_));
        debug_dst_pub_->publish(pclToPclRos(loop_closure_->getTargetCloud(), map_frame_));
        debug_fine_aligned_pub_->publish(pclToPclRos(loop_closure_->getFinalAlignedCloud(), map_frame_));
        debug_coarse_aligned_pub_->publish(pclToPclRos(loop_closure_->getCoarseAlignedCloud(), map_frame_));
    }
    loop_added_flag_vis_ = loop_added_flag_; // Signal vis timer to update
}

void FastLioSamScQn::visTimerFunc()
{
    if (!is_initialized_)
    {
        return;
    }

    high_resolution_clock::time_point tv1 = high_resolution_clock::now();
    //// 1. if loop closed, correct vis data
    if (loop_added_flag_vis_)
    // copy and ready
    {
        gtsam::Values corrected_esti_copied;
        pcl::PointCloud<pcl::PointXYZ> corrected_odoms;
        nav_msgs::msg::Path corrected_path;
        {
            std::lock_guard<std::mutex> lock(realtime_pose_mutex_);
            corrected_esti_copied = corrected_esti_;
        }
        // correct pose and path
        for (size_t i = 0; i < corrected_esti_copied.size(); ++i)
        {
            gtsam::Pose3 pose_ = corrected_esti_copied.at<gtsam::Pose3>(i);
            corrected_odoms.points.emplace_back(pose_.translation().x(), pose_.translation().y(), pose_.translation().z());
            corrected_path.poses.push_back(gtsamPoseToPoseStamped(pose_, map_frame_));
        }
        // update vis of loop constraints
        if (!loop_idx_pairs_.empty())
        {
            loop_detection_pub_->publish(getLoopMarkers(corrected_esti_copied));
        }
        // update with corrected data
        {
            std::lock_guard<std::mutex> lock(vis_mutex_);
            corrected_odoms_ = corrected_odoms;
            corrected_path_.poses = corrected_path.poses;
        }
        loop_added_flag_vis_ = false;
    }
    //// 2. publish odoms, paths
    {
        std::lock_guard<std::mutex> lock(vis_mutex_);
        odom_pub_->publish(pclToPclRos(odoms_, map_frame_));
        path_pub_->publish(odom_path_);
        corrected_odom_pub_->publish(pclToPclRos(corrected_odoms_, map_frame_));
        corrected_path_pub_->publish(corrected_path_);
    }

    //// 3. global map
    if (global_map_vis_switch_ && corrected_pcd_map_pub_->get_subscription_count() > 0) // save time, only once
    {
        pcl::PointCloud<PointType>::Ptr corrected_map(new pcl::PointCloud<PointType>());
        if (keyframes_.empty()) return;
        corrected_map->reserve(keyframes_[0].pcd_.size() * keyframes_.size()); // it's an approximated size
        {
            std::lock_guard<std::mutex> lock(keyframes_mutex_);
            for (size_t i = 0; i < keyframes_.size(); ++i)
            {
                *corrected_map += transformPcd(keyframes_[i].pcd_, keyframes_[i].pose_corrected_eig_);
            }
        }
        const auto &voxelized_map = voxelizePcd(corrected_map, voxel_res_);
        corrected_pcd_map_pub_->publish(pclToPclRos(*voxelized_map, map_frame_));
        global_map_vis_switch_ = false;
    }
    if (!global_map_vis_switch_ && corrected_pcd_map_pub_->get_subscription_count() == 0)
    {
        global_map_vis_switch_ = true;
    }
    high_resolution_clock::time_point tv2 = high_resolution_clock::now();

    // Not used Log
    // RCLCPP_INFO(this->get_logger(), "vis: %.1fms",
    //            duration_cast<microseconds>(tv2 - tv1).count() / 1e3);
    return;
}

void FastLioSamScQn::saveFlagCallback(const std_msgs::msg::String::ConstSharedPtr &msg)
{
    std::string save_dir = msg->data != "" ? msg->data : package_path_;

    // save scans as individual pcd files and poses in KITTI format
    // Delete the scans folder if it exists and create a new one
    std::string seq_directory = save_dir + "/" + seq_name_;
    std::string scans_directory = seq_directory + "/scans";
    if (save_in_kitti_format_)
    {
        RCLCPP_INFO(
            this->get_logger(),
            "\033[32;1mScans are saved in %s, following the KITTI and TUM "
            "format\033[0m",
            scans_directory.c_str());
        if (fs::exists(seq_directory))
        {
            fs::remove_all(seq_directory);
        }
        fs::create_directories(scans_directory);

        std::ofstream kitti_pose_file(seq_directory + "/poses_kitti.txt");
        std::ofstream tum_pose_file(seq_directory + "/poses_tum.txt");
        tum_pose_file << "#timestamp x y z qx qy qz qw\n";
        {
            std::lock_guard<std::mutex> lock(keyframes_mutex_);
            for (size_t i = 0; i < keyframes_.size(); ++i)
            {
                // Save the point cloud
                std::stringstream ss_;
                ss_ << scans_directory << "/" << std::setw(6) << std::setfill('0') << i << ".pcd";
                RCLCPP_INFO(this->get_logger(), "Saving %s...", ss_.str().c_str());
                pcl::io::savePCDFileASCII<PointType>(ss_.str(), keyframes_[i].pcd_);

                // Save the pose in KITTI format
                const auto &pose_ = keyframes_[i].pose_corrected_eig_;
                kitti_pose_file << pose_(0, 0) << " " << pose_(0, 1) << " " << pose_(0, 2) << " "
                                << pose_(0, 3) << " " << pose_(1, 0) << " " << pose_(1, 1) << " "
                                << pose_(1, 2) << " " << pose_(1, 3) << " " << pose_(2, 0) << " "
                                << pose_(2, 1) << " " << pose_(2, 2) << " " << pose_(2, 3) << "\n";

                const auto &lidar_optim_pose_ = poseEigToPoseStamped(keyframes_[i].pose_corrected_eig_);
                tum_pose_file << std::fixed << std::setprecision(8) << keyframes_[i].timestamp_
                              << " " << lidar_optim_pose_.pose.position.x << " "
                              << lidar_optim_pose_.pose.position.y << " "
                              << lidar_optim_pose_.pose.position.z << " "
                              << lidar_optim_pose_.pose.orientation.x << " "
                              << lidar_optim_pose_.pose.orientation.y << " "
                              << lidar_optim_pose_.pose.orientation.z << " "
                              << lidar_optim_pose_.pose.orientation.w << "\n";
            }
        }
        kitti_pose_file.close();
        tum_pose_file.close();
        RCLCPP_INFO(
            this->get_logger(),
            "\033[32;1mScans and poses saved in .pcd and KITTI format\033[0m");
    }

    if (save_map_pcd_)
    {
        pcl::PointCloud<PointType>::Ptr corrected_map(new pcl::PointCloud<PointType>());
        if (keyframes_.empty()) return;
        corrected_map->reserve(keyframes_[0].pcd_.size() * keyframes_.size()); // it's an approximated size
        {
            std::lock_guard<std::mutex> lock(keyframes_mutex_);
            for (size_t i = 0; i < keyframes_.size(); ++i)
            {
                *corrected_map += transformPcd(keyframes_[i].pcd_, keyframes_[i].pose_corrected_eig_);
            }
        }
        const auto &voxelized_map = voxelizePcd(corrected_map, voxel_res_);
        pcl::io::savePCDFileASCII<PointType>(seq_directory + "/" + seq_name_ + "_map.pcd", *voxelized_map);
        RCLCPP_INFO(
            this->get_logger(),
            "\033[32;1mAccumulated map cloud saved in .pcd format\033[0m");
    }
}

FastLioSamScQn::~FastLioSamScQn()
{

}

void FastLioSamScQn::updateOdomsAndPaths(const PosePcd &pose_pcd_in)
{
    odoms_.points.emplace_back(pose_pcd_in.pose_eig_(0, 3),
                               pose_pcd_in.pose_eig_(1, 3),
                               pose_pcd_in.pose_eig_(2, 3));
    corrected_odoms_.points.emplace_back(pose_pcd_in.pose_corrected_eig_(0, 3),
                                         pose_pcd_in.pose_corrected_eig_(1, 3),
                                         pose_pcd_in.pose_corrected_eig_(2, 3));
    odom_path_.poses.emplace_back(poseEigToPoseStamped(pose_pcd_in.pose_eig_, map_frame_));
    corrected_path_.poses.emplace_back(poseEigToPoseStamped(pose_pcd_in.pose_corrected_eig_, map_frame_));
    return;
}

visualization_msgs::msg::Marker FastLioSamScQn::getLoopMarkers(const gtsam::Values &corrected_esti_in)
{
    visualization_msgs::msg::Marker edges;
    edges.type = 5u;
    edges.scale.x = 0.12f;
    edges.header.frame_id = map_frame_;
    edges.pose.orientation.w = 1.0f;
    edges.color.r = 1.0f;
    edges.color.g = 1.0f;
    edges.color.b = 1.0f;
    edges.color.a = 1.0f;
    for (size_t i = 0; i < loop_idx_pairs_.size(); ++i)
    {
        if (loop_idx_pairs_[i].first >= corrected_esti_in.size() ||
            loop_idx_pairs_[i].second >= corrected_esti_in.size())
        {
            continue;
        }
        gtsam::Pose3 pose = corrected_esti_in.at<gtsam::Pose3>(loop_idx_pairs_[i].first);
        gtsam::Pose3 pose2 = corrected_esti_in.at<gtsam::Pose3>(loop_idx_pairs_[i].second);
        geometry_msgs::msg::Point p, p2;
        p.x = pose.translation().x();
        p.y = pose.translation().y();
        p.z = pose.translation().z();
        p2.x = pose2.translation().x();
        p2.y = pose2.translation().y();
        p2.z = pose2.translation().z();
        edges.points.push_back(p);
        edges.points.push_back(p2);
    }
    return edges;
}

bool FastLioSamScQn::checkIfKeyframe(const PosePcd &pose_pcd_in, const PosePcd &latest_pose_pcd)
{
    return keyframe_thr_ < (latest_pose_pcd.pose_corrected_eig_.block<3, 1>(0, 3) - pose_pcd_in.pose_corrected_eig_.block<3, 1>(0, 3)).norm();
}
