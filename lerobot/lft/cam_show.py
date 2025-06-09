import cv2

# 打开两个相机
cap0 = cv2.VideoCapture(0)
cap1 = cv2.VideoCapture(2)

if not cap0.isOpened():
    print("无法打开 /dev/video0")
if not cap1.isOpened():
    print("无法打开 /dev/video1")

while cap0.isOpened() and cap1.isOpened():
    ret0, frame0 = cap0.read()
    ret1, frame1 = cap1.read()

    if not ret0 or not ret1:
        print("读取图像失败")
        break

    # 可选：调整图像大小使得并排显示更合适
    frame0 = cv2.resize(frame0, (640, 480))
    frame1 = cv2.resize(frame1, (640, 480))

    # 拼接两个图像
    combined = cv2.hconcat([frame0, frame1])

    # 显示图像
    cv2.imshow("Camera 0 and Camera 1", combined)

    # 按 q 键退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放资源
cap0.release()
cap1.release()
cv2.destroyAllWindows()
