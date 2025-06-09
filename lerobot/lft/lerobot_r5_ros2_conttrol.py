#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from bimanual import SingleArm
import numpy as np

class ShoubingArmController(Node):
    def __init__(self):
        super().__init__('shoubing_arm_controller')

        # Initialize the arm
        arm_config = {
            "can_port": "can1",
            "type": 0,
        }
        self.single_arm = SingleArm(arm_config)
        self.prev_position = None
        self.allow_control = False

        # Subscribe to joint_position control topic
        self.joint_cmd_sub = self.create_subscription(
            JointState,
            '/master/joint_right',
            self.joint_position_callback,
            10
        )

        # Create publisher for joint states
        self.joint_state_pub = self.create_publisher(
            JointState,
            '/puppet/joint_right',
            10
        )

        # Create a timer for publishing joint states at 50Hz
        timer_period = 1.0 / 50.0  # seconds
        self.pub_timer = self.create_timer(timer_period, self.publish_joint_states)

        # self.home_pose = np.array([0.003, 0.046, 0.169, 0.003, 1.031, 0.273])
        # self.single_arm.set_ee_pose_xyzrpy(self.home_pose)

        self.get_logger().info("Shoubing control node started, waiting for commands...")

    def lerp(start, end, alpha):
        return start + alpha * (end - start)

    def joint_position_callback(self, msg):
        if len(msg.position) >= 7:
            target_joints = np.array(msg.position[:6])
            target_catch = msg.position[6]

            self.get_logger().info(f"Received joint command: {target_joints}, catch: {target_catch}")

            # 当前关节状态
            current_joints = np.array(self.single_arm.get_joint_positions())  # 你需要有这个方法
            current_catch = self.single_arm.get_catch_pos()  # 你需要有这个方法

            steps = 10  # 插值步数，越大越平滑越慢
            for i in range(1, steps + 1):
                alpha = i / steps
                interp_joints = lerp(current_joints, target_joints, alpha)
                interp_catch = lerp(current_catch, target_catch, alpha)

                self.single_arm.set_joint_positions(interp_joints.tolist())
                self.single_arm.set_catch_pos(interp_catch)

                rclpy.spin_once(self, timeout_sec=0.02)  # 20ms 间隔，约 50Hz（根据需要调整）

        else:
            self.get_logger().warn("Received joint command with fewer than 7 positions, ignoring")

    def publish_joint_states(self):
        # Get current joint positions
        joint_positions = self.single_arm.get_joint_positions()
        print("joint_positions:", joint_positions)

        # Create and fill JointState message
        joint_state_msg = JointState()
        joint_state_msg.header.stamp = self.get_clock().now().to_msg()
        joint_state_msg.name = [f"joint_{i}" for i in range(7)]
        joint_state_msg.position = joint_positions[:7]

        # Publish the message
        self.joint_state_pub.publish(joint_state_msg)

        # Optional: log at ~1Hzy
        now_sec = self.get_clock().now().seconds_nanoseconds()[0]
        if now_sec % 1 == 0:
            self.get_logger().info(f"Published joint states: {joint_positions[:7]}")
    

def main(args=None):
    rclpy.init(args=args)
    controller = ShoubingArmController()
    rclpy.spin(controller)
    controller.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
