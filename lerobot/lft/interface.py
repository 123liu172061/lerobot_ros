
import time
import torch
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from threading import Thread
from sensor_msgs.msg import JointState

from lerobot.common.policies.factory import make_policy
from lerobot.common.robot_devices.robots.manipulator import ManipulatorRobotConfig
#from lerobot.common.robot_devices.cameras.opencv import OpenCVCamera, OpenCVCameraConfig
from lerobot.common.robot_devices.cameras.configs import (
    CameraConfig,
    IntelRealSenseCameraConfig,
    OpenCVCameraConfig,
)
from lerobot.common.robot_devices.cameras.utils import make_cameras_from_configs

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.robot_devices.control_utils import init_keyboard_listener
from lerobot.configs.policies import PreTrainedConfig
from lerobot.common.utils.utils import get_safe_torch_device, has_method


import rerun as rr

from dataclasses import dataclass, field
from rclpy.task import Future
import os
import yaml
from typing import Optional
from pathlib import Path
import shutil
from copy import copy
from contextlib import nullcontext

# 机器人配置
@dataclass
class SevenDOFRobotConfig(ManipulatorRobotConfig):
    """7自由度单臂机器人的配置，使用 ROS 2 话题和 USB 相机进行数据采集。"""
    leader_arms: dict[str, list[str]] = field(
        default_factory=lambda: {
            "arm": [
                "shoulder_pan",
                "shoulder_lift",
                "elbow_flex",
                "wrist_1",
                "wrist_2",
                "wrist_3",
                "gripper",
            ]
        }
    )
    follower_arms: dict[str, list[str]] = field(
        default_factory=lambda: {
            "arm": [
                "shoulder_pan",
                "shoulder_lift",
                "elbow_flex",
                "wrist_1",
                "wrist_2",
                "wrist_3",
                "gripper",
            ]
        }
    )

    cameras: dict[str, CameraConfig] = field(
        default_factory=lambda: {
            "laptop": OpenCVCameraConfig(
                camera_index=0,
                fps=30,
                width=640,
                height=480,
            ),
            "phone": OpenCVCameraConfig(
                camera_index=2,
                fps=30,
                width=640,
                height=480,
            ),
        }
    )

    mock: bool = False


#任务配置
@dataclass
class RecordControlConfig:
    repo_id: str
    fps: int
    num_episodes: int
    single_task: str
    root: str = "./"
    warmup_time_s: float = 5.0
    episode_time_s: float = 60.0
    reset_time_s: float = 5.0
    resume: bool = False  # 默认创建新数据集
    video: bool = True   # 保存视频
    num_image_writer_processes: int = 0
    num_image_writer_threads_per_camera: int = 4
    display_data: bool = False  # 可视化
    policy: Optional[str] = None
    push_to_hub: bool = False
    tags: Optional[list] = None
    private: bool = False
    play_sounds: bool = True  #语音
    


# 机器人获取数据实现
class ROS2Robot(Node):
    def __init__(self, config: SevenDOFRobotConfig):
        super().__init__("lerobot_control")

        self.config = config  # 这一行非常关键！
        self.robot_type = "seven_dof"
        self.leader_arms = list(config.leader_arms.keys())  # ["arm"]
        self.follower_arms = list(config.follower_arms.keys())  # ["arm"]
        self.joint_names = config.leader_arms["arm"]  # 7 个关节
        print("self.config.cameras",self.config.cameras)
        self.cameras = make_cameras_from_configs(self.config.cameras)

        self.logs = {}
        self.is_connected = False
        self.callback_group = ReentrantCallbackGroup()


    #camera
    @property
    def camera_features(self) -> dict:
        cam_ft = {}
        for cam_key, cam in self.cameras.items():
            key = f"observation.images.{cam_key}"
            cam_ft[key] = {
                "shape": (cam.height, cam.width, cam.channels),
                "names": ["height", "width", "channels"],
                "info": None,
            }
        return cam_ft
    
    @property
    def motor_features(self) -> dict:
        action_names = [
                "shoulder_pan",
                "shoulder_lift",
                "elbow_flex",
                "wrist_1",
                "wrist_2",
                "wrist_3",
                "gripper",
        ]
        state_names = [
                "shoulder_pan",
                "shoulder_lift",
                "elbow_flex",
                "wrist_1",
                "wrist_2",
                "wrist_3",
                "gripper",
        ]

        return {
            "action": {
                "dtype": "float32",
                "shape": (len(action_names),),
                "names": action_names,
            },
            "observation.state": {
                "dtype": "float32",
                "shape": (len(state_names),),  
                "names": (state_names),
            },
        }
        
    @property
    def has_camera(self):
        return len(self.cameras) > 0
    @property
    def num_cameras(self):
        return len(self.cameras)


    def connect(self):
        if self.is_connected:
            return
        self.is_connected = True
        for name in self.cameras:
            self.cameras[name].connect()

    def disconnect(self):
        if not self.is_connected:
            raise RobotDeviceNotConnectedError(
                "ManipulatorRobot is not connected. You need to run `robot.connect()` before disconnecting."
            )
        for name in self.cameras:
            self.cameras[name].disconnect()

    # 核心数据通信
    def wait_for_message(self, topic, msg_type, timeout):
        future = Future()
        sub = self.create_subscription(
            msg_type, topic, lambda msg: future.set_result(msg), 10, callback_group=self.callback_group
        )
        try:
            rclpy.spin_until_future_complete(self, future, timeout_sec=timeout)
            if future.done():
                return future.result()
            raise TimeoutError(f"Timeout waiting for message on {topic}")
        finally:
            self.destroy_subscription(sub)

    # 核心数据通信
    def publish_once_message(self, topic, msg_type, msg, delay_sec=0.1):
        done = Future()

        pub = self.create_publisher(msg_type, topic, 10)

        def timer_callback():
            pub.publish(msg)
            self.get_logger().info(f"Published message once to {topic}")
            done.set_result(True)

        timer = self.create_timer(delay_sec, timer_callback)

        try:
            rclpy.spin_until_future_complete(self, done, timeout_sec=delay_sec + 1.0)
            if not done.done():
                raise TimeoutError(f"Timeout publishing message to {topic}")
        finally:
            self.destroy_timer(timer)
            self.destroy_publisher(pub)


    # 遥操作部分,推理脚本中无用
    def teleop_step(
        self, record_data=False
    ) -> None | tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        if not self.is_connected:
            raise RobotDeviceNotConnectedError(
                "ROSRobot is not connected. You need to run `robot.connect()`."
            )

        if not record_data:
            return

        leader_pos = {}
        for name in self.leader_arms:
            before_lread_t = time.perf_counter()
            try:
                msg = self.wait_for_message("/joint_states_target", JointState, timeout=0.1)
                print("leader_pos position:", msg.position)#有值
                print("name:", name)
                # 直接按顺序存，不管 joint name
                leader_pos[name] = torch.tensor(msg.position, dtype=torch.float32)
                print("leader_pos[name]:", leader_pos[name]) 

            except TimeoutError:
                self.get_logger().warn(f"Failed to read /joint_states_target for {name}")
                leader_pos[name] = torch.zeros(len(self.joint_names), dtype=torch.float32)

            self.logs[f"read_leader_{name}_pos_dt_s"] = time.perf_counter() - before_lread_t


        # 从 /joint_states_now 读取当前位置（状态）
        follower_pos = {}
        for name in self.follower_arms:
            before_fread_t = time.perf_counter()
            try:
                msg = self.wait_for_message("/joint_states_now", JointState, timeout=0.1)
                follower_pos[name] = torch.tensor(msg.position, dtype=torch.float32)
            except TimeoutError:
                self.get_logger().warn(f"Failed to read /joint_states_now for {name}")
                follower_pos[name] = torch.zeros(len(self.joint_names), dtype=torch.float32)
            self.logs[f"read_follower_{name}_pos_dt_s"] = time.perf_counter() - before_fread_t

        # 拼接从臂当前位置作为状态
        # 我直接一次性读了7个关节，应该不需要拼接
        state = []
        for name in self.follower_arms:
            if name in follower_pos:
                state.append(follower_pos[name])
        state = torch.cat(state) if state else torch.tensor([], dtype=torch.float32)
        print("state:", state)

        # 拼接主臂目标位置作为动作
        action = []
        for name in self.leader_arms:
            if name in leader_pos:
                action.append(leader_pos[name])
        action = torch.cat(action) if action else torch.tensor([], dtype=torch.float32)
        print("action:", action)

        # 使用 USB 相机捕获图像
        images = {}
        for name in self.cameras:
            before_camread_t = time.perf_counter()
            img = self.cameras[name].read()
            images[name] = torch.from_numpy(img) if img is not None else torch.zeros(480, 640, 3, dtype=torch.uint8)
            self.logs[f"read_camera_{name}_dt_s"] = 1.0 / self.cameras[name].fps
            self.logs[f"read_camera_{name}_dt_s"] = time.perf_counter() - before_camread_t

        # 构建输出字典
        obs_dict, action_dict = {}, {}
        obs_dict["observation.state"] = state
        action_dict["action"] = action
        for name in self.cameras:
            obs_dict[f"observation.images.{name}"] = images[name]

        return obs_dict, action_dict

    def capture_observation(self):

        # 获取机械臂当前关节位置
        before_fread_t = time.perf_counter()
        msg = self.wait_for_message("/puppet/joint_right", JointState, timeout=5.0)
            # 检查消息时间是否“最近”
        from rclpy.time import Time  # ROS2 的时间类
        msg_time = Time.from_msg(msg.header.stamp)
        now_time = self.get_clock().now()
        age = (now_time - msg_time).nanoseconds * 1e-9  # 秒
        if age > 0.1:
            self.get_logger().warn(f"JointState message is too old: {age:.3f}s ago")

        position = torch.tensor(msg.position, dtype=torch.float32)
        state = position
        self.logs[f"read_follower_pos_dt_s"] = time.perf_counter() - before_fread_t

        print("state:",state)

        # Capture images from cameras
        images = {}
        for name in self.cameras:
            before_camread_t = time.perf_counter()
            images[name] = self.cameras[name].async_read()
            images[name] = torch.from_numpy(images[name])
            self.logs[f"read_camera_{name}_dt_s"] = self.cameras[name].logs["delta_timestamp_s"]
            print("read_camera_{name}:",self.cameras[name].logs["delta_timestamp_s"]) # 图像与当前时间的差值
            self.logs[f"async_read_camera_{name}_dt_s"] = time.perf_counter() - before_camread_t
            print("async_read_camera_{name}:",self.cameras[name].logs["delta_timestamp_s"])  #调用耗时

        obs_dict = {"observation.state": state}
        for name in self.cameras:
            obs_dict[f"observation.images.{name}"] = images[name]
        print("obs_dict:",obs_dict)

        return obs_dict

    def send_action(self, action: torch.Tensor) -> torch.Tensor:
        if not self.is_connected:
            raise RobotDeviceNotConnectedError("ManipulatorRobot is not connected.")

        joint_names = self.joint_names  # ["joint_0", ..., "joint_6"]

        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()  # ROS 2 时间戳
        msg.name = joint_names
        msg.position = action.tolist()

        self.publish_once_message("/master/joint_right", JointState, msg)

        return action


def predict_action(observation, policy, device, use_amp):
    observation = copy(observation)
    with (
        torch.inference_mode(),
        torch.autocast(device_type=device.type) if device.type == "cuda" and use_amp else nullcontext(),
    ):
        # Convert to pytorch format: channel first and float32 in [0,1] with batch dimension
        for name in observation:
            if "image" in name:
                observation[name] = observation[name].type(torch.float32) / 255
                observation[name] = observation[name].permute(2, 0, 1).contiguous()
            observation[name] = observation[name].unsqueeze(0)
            observation[name] = observation[name].to(device)

        # Compute the next action with the policy
        # based on the current observation
        action = policy.select_action(observation)

        # Remove batch dimension
        action = action.squeeze(0)

        # Move to cpu, if not already the case
        action = action.to("cpu")

    return action

# 辅助函数（简化版，模拟 LeRobot 行为）
def sanity_check_dataset_name(repo_id: str, policy: Optional[str]):
    if not repo_id:
        raise ValueError("repo_id cannot be empty")
    if policy and "invalid" in policy:
        raise ValueError("Invalid policy name")

def sanity_check_dataset_robot_compatibility(dataset: LeRobotDataset, robot: ROS2Robot, fps: int, video: bool):
    if dataset.fps != fps:
        raise ValueError(f"Dataset FPS ({dataset.fps}) does not match requested FPS ({fps})")
    if video and not robot.cameras:
        raise ValueError("Video recording requested but no cameras available")

def log_say(message: str, play_sounds: bool, blocking: bool = False):
    print(message)

def warmup_record(robot: ROS2Robot, events: dict, teleoperate: bool, warmup_time_s: float, display_data: bool, fps: int):
    start_time = time.perf_counter()
    while time.perf_counter() - start_time < warmup_time_s:
        if teleoperate:
            robot.teleop_step(record_data=False)
        if fps:
            time.sleep(1 / fps)
        if events.get("exit_early", False):
            break

def record_episode(
    robot: ROS2Robot,
    dataset: LeRobotDataset,
    events: dict,
    episode_time_s: float,
    display_data: bool,
    policy=None,
    fps: int = None,
    single_task: str = None,
):
    timestamp = 0
    start_episode_t = time.perf_counter()
    while timestamp < episode_time_s:
        start_loop_t = time.perf_counter()

        time_inter=time.perf_counter()
        if policy is not None:#如果有配置网络则进行推理
            # 网络推理
            observation = robot.capture_observation() 
            print("time_inter1:", time.perf_counter()-time_inter)
            pred_action = predict_action(
                observation, policy, get_safe_torch_device(policy.config.device), policy.config.use_amp
            )
            print("time_inter2:", time.perf_counter()-time_inter)
            # Action can eventually be clipped using `max_relative_target`,
            # so action actually sent is saved in the dataset.
            action = robot.send_action(pred_action)
            action = {"action": action}
            print("time_inter3:", time.perf_counter()-time_inter)
        #否则采数据，其实下面可以删掉，冗余了，因为采数据和推理脚本已经分开了
        else:
            observation, action = robot.teleop_step(record_data=True)
            if dataset is not None:
                frame = {**observation, **action, "task": single_task}
                print("frame:",frame)
                print("frame keys:", frame.keys())
                dataset.add_frame(frame)

        # 可视化
        if (display_data) or (display_data and robot.robot_type.startswith("lekiwi")):
            for k, v in action.items():
                for i, vv in enumerate(v):
                    rr.log(f"sent_{k}_{i}", rr.Scalar(vv.numpy()))

            image_keys = [key for key in observation if "image" in key]
            for key in image_keys:
                rr.log(key, rr.Image(observation[key].numpy()), static=True)
        
        
        
        if fps is not None:
            dt_s = time.perf_counter() - start_loop_t
            print("dt_s:", dt_s)
            if dt_s < 1 / fps:
                time.sleep(1 / fps - dt_s)

        print("time_inter4:", time.perf_counter()-time_inter)
        timestamp = time.perf_counter() - start_episode_t
        print("timestamp:", timestamp)
        if events.get("exit_early", False):
            events["exit_early"] = False
            break

def reset_environment(robot: ROS2Robot, events: dict, reset_time_s: float, fps: int):
    start_time = time.perf_counter()
    while time.perf_counter() - start_time < reset_time_s:
        robot.teleop_step(record_data=False)
        if fps:
            time.sleep(1 / fps)
        if events.get("exit_early", False):
            break

def stop_recording(robot: ROS2Robot, listener, display_data: bool):
    robot.disconnect()

# 录制函数（基于 LeRobot 的 record）
def record(
    robot: ROS2Robot,
    cfg: RecordControlConfig,
) -> LeRobotDataset:
    # 检查目录和元数据文件
    if os.path.exists(cfg.root):
        meta_info_path = os.path.join(cfg.root, "meta", "info.json")
        if cfg.resume and not os.path.exists(meta_info_path):
            print(f"警告：{meta_info_path} 不存在，无法恢复数据集。将删除 {cfg.root} 并创建新数据集。")
            shutil.rmtree(cfg.root)
            cfg.resume = False
        elif not cfg.resume:
            while True:
                print(f"目录 {cfg.root} 已存在。是否删除并创建新数据集？(y/n)")
                choice = input().strip().lower()
                if choice in ["y", "n"]:
                    break
                print("请输入 'y' 或 'n'")
            if choice == "y":
                shutil.rmtree(cfg.root)
            else:
                print("请删除现有目录或启用 resume=True")
                raise FileExistsError(f"目录 {cfg.root} 已存在")
    # cfg.resume=True：从已存在数据集恢复录制，不覆盖原有数据
    if cfg.resume:
        dataset = LeRobotDataset(
            cfg.repo_id,
            root=cfg.root,
            revision=None,  # 跳过远程版本检查
        )
        if len(robot.cameras) > 0:
            dataset.start_image_writer(
                num_processes=cfg.num_image_writer_processes,
                num_threads=cfg.num_image_writer_threads_per_camera * len(robot.cameras),
            )
        sanity_check_dataset_robot_compatibility(dataset, robot, cfg.fps, cfg.video)
    else:
        # 创建一个新的数据集目录（如果目录存在则加载已保存 episode）。
        # sanity_check_dataset_name(cfg.repo_id, cfg.policy)
        dataset = LeRobotDataset.create(
            cfg.repo_id,
            cfg.fps,
            root=cfg.root,
            robot=robot,
            use_videos=cfg.video,
            image_writer_processes=cfg.num_image_writer_processes,
            image_writer_threads=cfg.num_image_writer_threads_per_camera * len(robot.cameras),
        )

    print("cfg.policy =", cfg.policy)
    policy = None if cfg.policy is None else make_policy(cfg.policy, ds_meta=dataset.meta)

    if not robot.is_connected:
        robot.connect()

    listener, events = init_keyboard_listener()

    log_say("Warmup record", cfg.play_sounds)
    warmup_record(robot, events, policy is None, cfg.warmup_time_s, cfg.display_data, cfg.fps)

    recorded_episodes = 0
    while True:
        if recorded_episodes >= cfg.num_episodes:
            break

        log_say(f"Recording episode {dataset.num_episodes}", cfg.play_sounds)
        record_episode(
            robot=robot,
            dataset=dataset,
            events=events,
            episode_time_s=cfg.episode_time_s,
            display_data=cfg.display_data,
            policy=policy,
            fps=cfg.fps,
            single_task=cfg.single_task,
        )

        if not events["stop_recording"] and (
            (recorded_episodes < cfg.num_episodes - 1) or events["rerecord_episode"]
        ):
            log_say("Reset the environment", cfg.play_sounds)
            #reset_environment(robot, events, cfg.reset_time_s, cfg.fps)

        if events["rerecord_episode"]:
            log_say("Re-record episode", cfg.play_sounds)
            events["rerecord_episode"] = False
            events["exit_early"] = False
            dataset.clear_episode_buffer()
            continue

        # dataset.save_episode()
        recorded_episodes += 1

        if events["stop_recording"]:
            break

    log_say("Stop recording", cfg.play_sounds, blocking=True)
    stop_recording(robot, listener, cfg.display_data)

    if cfg.push_to_hub:
        dataset.push_to_hub(tags=cfg.tags, private=cfg.private)

    log_say("Exiting", cfg.play_sounds)
    return dataset

# 主函数
def main():
    rclpy.init()
    try:
        with open("lerobot/lft/config_lft.yaml", "r") as f:
            config = yaml.safe_load(f)

        robot_cfg=SevenDOFRobotConfig()
        robot = ROS2Robot(robot_cfg)

        policy_path = config["policy"]["model_path"]
        policy_cfg = PreTrainedConfig.from_pretrained(policy_path)
        policy_cfg.pretrained_path = policy_path  # 手动加上字段以兼容 downstream 代码

        cfg = RecordControlConfig(
            repo_id=config["control"]["repo_id"],
            fps=config["control"]["fps"],
            num_episodes=config["control"]["num_episodes"],
            single_task=config["control"]["single_task"],
            root=config["control"]["repo_id"],
            episode_time_s=10.0,
            warmup_time_s=5.0,
            reset_time_s=5.0,
            resume=False,
            video=True,
            num_image_writer_processes=0,
            num_image_writer_threads_per_camera=4,
            display_data=False,
            policy=policy_cfg,  # 如果为None，则采集数据，若为policy_cfg，则为推理
            play_sounds=False,
        )

        record(robot, cfg)
        #print(f"数据集已保存至 {dataset.root_dir}")

    finally:
        rclpy.shutdown()

if __name__ == "__main__":
    main()
