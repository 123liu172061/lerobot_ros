
# 遥操录制数据
python lerobot/scripts/control_robot.py record \
    --robot.type=so100 \
    --control.type=record \
    --control.fps 30 \
    --control.repo_id=$USER/koch_pick_place_lego \
    --control.num_episodes=50 \
    --control.warmup_time_s=2 \
    --control.episode_time_s=30 \
    --control.reset_time_s=10



# 训练
python lerobot/scripts/train.py \
  --dataset.repo_id=/home/lft/workspace/python_project/VLA/lerobot/r5-test2 \
  --policy.type=act \
  --output_dir=outputs/train/r5_test2 \
  --job_name=act_r5_test2 \
  --policy.device=cuda \
  --wandb.enable=false


# 推理
python lerobot/lft/interface.py
