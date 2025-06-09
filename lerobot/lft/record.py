
import time
import torch
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from threading import Thread
from sensor_msgs.msg import JointState


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

import rerun as rr

from dataclasses import dataclass, field
from rclpy.task import Future
import os
import yaml
from typing import Optional
from pathlib import Path
import shutil

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

    @classmethod
    def from_dict(cls, config_dict):
        """将 YAML 配置文件中的字典转换为 SevenDOFRobotConfig。"""
        cameras = {
            name: OpenCVCameraConfig(**cam_config)
            for name, cam_config in config_dict.get("cameras", {}).items()
        }
        config_dict = config_dict.copy()
        config_dict["cameras"] = cameras
        return cls(**config_dict)

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


        # self.motor_features = {
        #     "observation.state": {"shape": (7,), "dtype": "float32"},
        #     "action": {"shape": (7,), "dtype": "float32"}
        # }
        # self.observation_space = {
        #     "observation.state": {"shape": (7,), "dtype": "float32"},
        #     **{name: feat for name, feat in self.camera_features.items()}  # 使用 main_camera
        # }
        # self.action_space = {
        #     "action": {"shape": (7,), "dtype": "float32"}
        # }
    

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
                "shape": (len(state_names),),  # 3 个位置 + 3 个速度 = 6 维
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

    def teleop_step(
        self, record_data=False
    ) -> None | tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        if not self.is_connected:
            raise RobotDeviceNotConnectedError(
                "ROSRobot is not connected. You need to run `robot.connect()`."
            )

        if not record_data:
            return

        # 从 /joint_states_target 读取目标位置（动作）
        # leader_pos = {}
        # for name in self.leader_arms:
        #     before_lread_t = time.perf_counter()
        #     try:
        #         msg = self.wait_for_message("/joint_states_target", JointState, timeout=0.1)
        #         if set(msg.name) != set(self.joint_names):
        #             print("leader_pos position:", msg.position)#有值
        #             pos_dict = dict(zip(msg.name, msg.position))
        #             ordered_pos = [pos_dict.get(joint, 0.0) for joint in self.joint_names]
        #             print("ordered_pos:",ordered_pos)  #全是0
        #             leader_pos[name] = torch.tensor(ordered_pos, dtype=torch.float32)
        #             print("leader_pos[name]:", leader_pos[name]) #全是0
        #         else:
        #             leader_pos[name] = torch.tensor(msg.position, dtype=torch.float32)
        #     except TimeoutError:
        #         self.get_logger().warn(f"Failed to read /joint_states_target for {name}")
        #         leader_pos[name] = torch.zeros(len(self.joint_names), dtype=torch.float32)
        #     self.logs[f"read_leader_{name}_pos_dt_s"] = time.perf_counter() - before_lread_t
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
            if dt_s < 1 / fps:
                time.sleep(1 / fps - dt_s)

        timestamp = time.perf_counter() - start_episode_t
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

def make_policy(policy_name: str, ds_meta: dict):
    return None

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
        sanity_check_dataset_name(cfg.repo_id, cfg.policy)
        dataset = LeRobotDataset.create(
            cfg.repo_id,
            cfg.fps,
            root=cfg.root,
            robot=robot,
            use_videos=cfg.video,
            image_writer_processes=cfg.num_image_writer_processes,
            image_writer_threads=cfg.num_image_writer_threads_per_camera * len(robot.cameras),
        )

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
            reset_environment(robot, events, cfg.reset_time_s, cfg.fps)

        if events["rerecord_episode"]:
            log_say("Re-record episode", cfg.play_sounds)
            events["rerecord_episode"] = False
            events["exit_early"] = False
            dataset.clear_episode_buffer()
            continue

        dataset.save_episode()
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
        with open("/home/lft/workspace/python_project/VLA/lerobot/lerobot/lft/config_lft.yaml", "r") as f:
            config = yaml.safe_load(f)

        robot_cfg=SevenDOFRobotConfig()
        robot = ROS2Robot(robot_cfg)

        cfg = RecordControlConfig(
            repo_id=config["control"]["repo_id"],
            fps=config["control"]["fps"],
            num_episodes=config["control"]["num_episodes"],
            single_task=config["control"]["single_task"],
            root=config["control"]["repo_id"],
            episode_time_s=10.0,
            warmup_time_s=5.0,
            reset_time_s=5.0,
            resume=False,  # 默认创建新数据集
            video=True,
            num_image_writer_processes=0,
            num_image_writer_threads_per_camera=4,
            display_data=False,
            policy=None,
            push_to_hub=False,
            play_sounds=True, 
        )

        record(robot, cfg)
        #print(f"数据集已保存至 {dataset.root_dir}")

    finally:
        rclpy.shutdown()

if __name__ == "__main__":
    main()
