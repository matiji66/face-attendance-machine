import cv2
import time

"""
下面是从摄像头捕捉实时流并将其写入文件的Python实现。
运行程序后 按键Q推出，按键C 进行拍照并保存到当前的路径
"""

# Create a VideoCapture object
cap = cv2.VideoCapture(0)

# Default resolutions of the frame are obtained.The default resolutions are system dependent.
# We convert the resolutions from float to integer.
# 默认分辨率取决于系统。
# 我们将分辨率从float转换为整数。
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

# Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
# 定义编解码器并创建VideoWriter对象。输出存储在“outpy.avi”文件中。

out = cv2.VideoWriter('outpy{}.avi'.format(int(time.time())), cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30.0,
                      (frame_width, frame_height))
image_name = "image_{}.jpg"
index = 0
flag = True
# Check if camera opened successfully
while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        # Write the frame into the file 'output.avi'
        out.write(frame)
        # Display the resulting frame
        cv2.imshow('frame', frame)
        key = cv2.waitKey(1)
        # Press Q on keyboard to stop recordingqc
        if key & 0xFF == ord('q'):
            break
        if key & 0xFF == ord('c'):
            index += 1
            cv2.imwrite(image_name.format(index), frame)
    # Break the loop
    else:
        break

# When everything done, release the video capture and video write objects
cap.release()
out.release()

# Closes all the frames
cv2.destroyAllWindows()
