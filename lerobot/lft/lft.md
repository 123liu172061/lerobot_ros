# 激活虚拟环境
conda activate lerobot

# 遥操录制数据
python lerobot/lft/record.py



# 训练
## ACT:
python lerobot/scripts/train.py \
  --dataset.repo_id=/home/lft/workspace/python_project/VLA/lerobot/r5-test2 \
  --policy.type=act \
  --output_dir=outputs/ACT/train/r5_test2 \
  --job_name=act_r5_test2 \
  --policy.device=cuda \
  --wandb.enable=false

## DP
python lerobot/scripts/train.py \
  --dataset.repo_id=/home/lft/workspace/python_project/VLA/lerobot/r5-test2 \
  --policy.type=diffusion \
  --output_dir=outputs/DP/train/r5_test2 \
  --job_name=DP_r5_test2 \
  --policy.device=cuda \
  --wandb.enable=false




# 推理
python lerobot/lft/interface.py


## 测试相机
python lerobot/lft/cam_show.py  #查看图像

python lerobot/lft/test_cam.py  #索引号


## 如何获取usb帧率：
sudo apt install v4l-utils
1. 查看支持的格式和帧率
v4l2-ctl -d /dev/video0 --list-formats-ext
