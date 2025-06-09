# import cv2

# cap = cv2.VideoCapture(0, cv2.CAP_V4L2)  # 使用 V4L2 后端（推荐）

# # 设置分辨率
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# # 设置帧率（必须配合 MJPG 才有效）
# cap.set(cv2.CAP_PROP_FPS, 60)

# # 关键：设置为 MJPG 格式
# cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

# # 检查设置是否成功
# actual_fps = cap.get(cv2.CAP_PROP_FPS)
# actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
# actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
# fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
# fourcc_str = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])

# print(f"当前设置: 分辨率={int(actual_width)}x{int(actual_height)}, 帧率={actual_fps:.2f}, FOURCC={fourcc_str}")

# # 测试读取图像
# ret, frame = cap.read()
# if ret:
#     print("成功读取一帧图像")
# else:
#     print("读取失败")

# cap.release()


import cv2

# 测试相机
for i in range(4):  # 假设最多4个video设备
    cap = cv2.VideoCapture(i)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)  # 试图设置为30帧
    # # 关键：设置为 MJPG 格式
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    if cap.isOpened():
        print(f"/dev/video{i} 打开成功")
        
        # 尝试读取一帧
        ret, frame = cap.read()
        if ret:
            print(f"/dev/video{i} 能够读取图像，分辨率: {frame.shape}")
            
            # 获取帧率
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps == 0:
                print(f"/dev/video{i} 获取帧率失败（可能设备未设置输出帧率）")
            else:
                print(f"/dev/video{i} 帧率: {fps:.2f} FPS")
        else:
            print(f"/dev/video{i} 打开但无法读取图像")
        
        cap.release()
    else:
        print(f"/dev/video{i} 打开失败")


# sudo apt install v4l-utils

# 查看支持的格式和帧率
# v4l2-ctl -d /dev/video0 --list-formats-ext

# 设置
# v4l2-ctl -d /dev/video0 --set-fmt-video=width=640,height=480,pixelformat=MJPG
# v4l2-ctl -d /dev/video0 --set-parm=30
