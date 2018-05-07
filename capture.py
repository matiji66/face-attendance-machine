# !/usr/bin/python3.6
# -*- coding: utf-8 -*-
# @author breeze

import cv2
import os

"""
下面是从摄像头捕捉实时流,以及采集照片,并将其写入文件的Python实现。

运行程序后:
1. 现在命令行输入采集照片人的姓名(拼音),如me
2. 选中摄像头框,并切换到英文输入法,按键Q推出，按键C 进行拍照并保存到指定的路径,
   此处avi文件保存在当前路径,每个人的照片保存在dataset单独的文件夹中
3. 当捕获的照片数量大于size(8),重复步骤1

"""

# Create a VideoCapture object
cap = cv2.VideoCapture(0)

# Check if camera opened successfully
if not cap.isOpened():
    print("Unable to read camera feed")

# Default resolutions of the frame are obtained.The default resolutions are system dependent.
# We convert the resolutions from float to integer.
# 默认分辨率取决于系统。
# 我们将分辨率从float转换为整数。
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

# Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
# 定义编解码器并创建VideoWriter对象。输出存储在“outpy.avi”文件中。

name = "me"
image_base_path = "./dataset"
out = cv2.VideoWriter(image_base_path +'/outpy.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (frame_width, frame_height))
if not os.path.exists(image_base_path ):
    os.makedirs(image_base_path)

index = 0  #
size = 8  # record 8 different image from different direction
flag = True


while True:
    ret, frame = cap.read()
    if ret:
        # Write the frame into the file 'output.avi'
        out.write(frame)
        if flag and index == 0:  # or index % size == 0
            flag = False
            name = input("############please input a name and then press enter key to continue ############:")

        # Display the resulting frame
        cv2.imshow('frame', frame)

        key = cv2.waitKey(1)
        # Press Q on keyboard to stop recordingqc
        if key & 0xFF == ord('q'):
            break
        if key & 0xFF == ord('c'):
            index += 1
            if not os.path.exists(image_base_path + "/" + name):
                os.makedirs(image_base_path + "/" + name)
            cv2.imwrite("{}/{}/{}_{}.jpg".format(image_base_path, name, name, index), frame)
            if index == size:
                index = 0
                flag = True

    # Break the loop
    else:
        break

# When everything done, release the video capture and video write objects
cap.release()
out.release()

# Closes all the frames
cv2.destroyAllWindows()

