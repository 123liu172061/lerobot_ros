import cv2
#测试相机
for i in range(4):  # 假设最多4个video设备
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f"/dev/video{i} 打开成功")
        ret, frame = cap.read()
        if ret:
            print(f"/dev/video{i} 能够读取图像，分辨率: {frame.shape}")
        else:
            print(f"/dev/video{i} 打开但无法读取图像")
        cap.release()
    else:
        print(f"/dev/video{i} 打开失败")
