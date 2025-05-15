#include "fast_lio_sam_sc_qn.h"

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<FastLioSamScQn>();

    rclcpp::executors::MultiThreadedExecutor executor;
    executor.add_node(node);

    // Spin the node until shutdown
    executor.spin();

    // Shutdown ROS2
    rclcpp::shutdown();

    return 0;
}
