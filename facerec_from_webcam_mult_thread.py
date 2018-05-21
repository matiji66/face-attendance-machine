# !/usr/bin/python3.6
# -*- coding: utf-8 -*-
# @author breeze
import threading
import argparse
import multiprocessing
import time
from multiprocessing import Queue, Pool

import face_recognition
import pandas as pd
import win32com.client
import cv2
import encoding_images
from app_utils import *

# This is a demo of running face recognition on live video from your webcam. It's a little more complicated than the
# other example, but it includes some basic performance tweaks to make things run a lot faster:
#   1. Process each video frame at 1/4 resolution (though still display it at full resolution)
#   2. Only detect faces in every other frame of video.

# PLEASE NOTE: This example requires OpenCV (the `cv2` library) to be installed only to read from your webcam.
# OpenCV is *not* required to use the face_recognition library. It's only required if you want to run this
# specific demo. If you have trouble installing it, try any of the other demos that don't require it instead.


# Load a sample picture and learn how to recognize it.
# face_recognition.api.batch_face_locations(images, number_of_times_to_upsample=1, batch_size=128)[source]
# face_recognition.api.compare_faces(known_face_encodings, face_encoding_to_check, tolerance=0.6)
# face_recognition.api.face_distance(face_encodings, face_to_compare)[source]
# face_recognition.api.face_encodings(face_image, known_face_locations=None, num_jitters=1)[source]
# face_recognition.api.face_landmarks(face_image, face_locations=None)[source]
# face_recognition.api.face_locations(img, number_of_times_to_upsample=1, model='hog')[source]
# face_recognition.api.load_image_file(file, mode='RGB')[source]


# 语音模块 voice model
speaker = win32com.client.Dispatch("SAPI.SpVoice")

name = "Unknown"
current_names = [name]
last_time = time.time()
known_face_names = []
known_face_encodings = []
known_face_encodings, known_face_names = encoding_images.load_encodings()

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True  #
TIME_DIFF = 20  # 持久化的时间间隔,当设置为 0 时候,每次识别的结果直接进行保存.
name_record = "./dataset/face_record.txt"  # 持久化识别出的人脸结果
NAME_DF = pd.DataFrame(known_face_names, columns=["name"])
last_ts = time.time()
lock = threading.Lock()


def myprint(log, ts):
    global lock, last_ts
    if lock.acquire():
        diff = ts - last_ts
        print(log, '--------', diff)
        last_ts = ts
        lock.release()


def process_face_records(name):
    """
    处理每一条识别的记录 ,并在一定时间之后将数据持久化到文件中
    此处会碰到全局并发,导致锁的问题
    :param name:
    :return:
    """
    return
    print('process_face_records start', time.time())

    global current_names, last_time
    # myprint("global current_names {}, last_time {}".format(current_names, last_time))

    # 判断是不是在识别的列表中,不在的话就进行问候
    if name not in current_names:
        print("ts ====", last_time, time.time())
        current_names.append(name)
        myprint("Hello {}, nice to meet you! ".format(name))
        # speaker.Speak("Hello {}, nice to meet you! ".format(name))

    # 在一定时间内,清空已经识别的人, 并进行
    if last_time < time.time() - TIME_DIFF:  # 每隔一段时间清空一下检测到的人
        last_time = time.time()
        time_format = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        myprint(time_format + " update last_time and clear current_names.")
        with open(name_record, 'a') as f:
            if len(current_names) > 0:
                f.writelines("{}:{} \n".format(time_format, str(current_names)))
        print("======", current_names)
        current_names = []  # clear()
        current_names = [name]
        myprint('process_face_records end', time.time())


def vote_class(face_encoding, tolerance=0.3, topN=5):
    myprint('vote start ', time.time())
    """
    当比较的结果小于tolerance的时候,有多个值,采用取topN 进行投票 ,决定最终的分类,此处没有对 distance 距离进行加权
    :param face_encoding: face encoding
    :param tolerance: 距离的阈值,越小越相似
    :param topN: 参与投票的最大数量
    :return: detect name
    """
    # 计算出距离
    distance_ = face_recognition.face_distance(known_face_encodings, face_encoding)
    df = pd.DataFrame(distance_, columns=["dis"])  # 转换成 DataFrame
    topDF = df[df['dis'] <= tolerance].nsmallest(topN, columns=['dis'])  # 过滤结果集
    namedf = NAME_DF.loc[topDF.index]  # 从姓名列表中获取face距离对应的人脸名称
    con = pd.concat([topDF, namedf], axis=1)  # concat name and distance
    # print('con', con)
    group = con.groupby(["name"])['dis'].sum()
    gp = group.reset_index()
    print('vote -- ', gp)
    if len(gp) == 0:
        print("------unknown -----")
        return "Unknown", 10
    import numpy as np  # TODO  optimize
    arr = np.array(gp)
    name1 = arr[0, 0]
    dis1 = arr[0, 1]
    print("get top one:", name1, dis1)
    myprint('vote end', time.time())
    return name1, dis1


def face_process(frame):
    # Resize frame of video to 1/4 size for faster face recognition processing
    myprint("face process resize start", time.time())
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    myprint("face process small_frame start", time.time())
    rgb_small_frame = small_frame[:, :, ::-1]

    # Find all the faces and face encodings in the current frame of video
    # face_locations = face_recognition.face_locations(rgb_small_frame, model="cnn")
    myprint('face_locations start', time.time())
    face_locations = face_recognition.face_locations(rgb_small_frame, model="hog")
    myprint('face_locations end', time.time())
    myprint('face_encodings start', time.time())
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
    myprint('face_encodings end', time.time())
    face_names = []
    for face_encoding in face_encodings:
        # optimize start 采用KNN 排名*权重, 在类别上进行叠加,然后排序取出top1
        name, dis = vote_class(face_encoding)
        # optimize end 采用 排名*权重, 在类别上进行叠加,然后排序取出top1
        face_names.append(name)  # 将人脸数据

    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4
        myprint('putText start', time.time())
        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
        myprint("putText end " + name, time.time())
        # say hello and save record to file
        myprint('process_face_records start', time.time())
        process_face_records(name)
        myprint('process_face_records end', time.time())

    # Display the resulting image
    # cv2.imshow('Video', frame)
    myprint("face process end", time.time())
    return frame


def worker(input_q, output_q):
    # Load a (frozen) Tensorflow model into memory.
    fps = FPS().start()
    while True:
        myprint("updata start ", time.time())
        fps.update()
        myprint("updata end ", time.time())
        # global lock
        # if lock.acquire():
        #    lock.release()

        frame = input_q.get()
        myprint("out queue {} and input que size {} after input_q get".format(output_q.qsize(), input_q.qsize()), time.time())
        myprint("out queue {} and input que size {} after lock release ".format(output_q.qsize(), input_q.qsize()), time.time())
        myprint("face process start", time.time())
        # frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        out_frame = face_process(frame)
        myprint("out queue {} and input que size {}".format(output_q.qsize(), input_q.qsize()), time.time())
        output_q.put(out_frame)
        myprint("out queue {} and input que size {} ".format(output_q.qsize(), input_q.qsize()), time.time())

    fps.stop()


if __name__ == '__main__':
    width = 640
    height = 480
    num_workers = 3
    queue_size = 5
    parser = argparse.ArgumentParser()
    parser.add_argument('-src', '--source', dest='video_source', type=int,
                        default=0, help='Device index of the camera.')
    parser.add_argument('-wd', '--width', dest='width', type=int,
                        default=width, help='Width of the frames in the video stream.')
    parser.add_argument('-ht', '--height', dest='height', type=int,
                        default=height, help='Height of the frames in the video stream.')
    parser.add_argument('-num-w', '--num-workers', dest='num_workers', type=int,
                        default=num_workers, help='Number of workers.')
    parser.add_argument('-q-size', '--queue-size', dest='queue_size', type=int,
                        default=queue_size, help='Size of the queue.')

    args = parser.parse_args()
    logger = multiprocessing.log_to_stderr()
    logger.setLevel(multiprocessing.SUBDEBUG)
    input_q = Queue(maxsize=args.queue_size)
    output_q = Queue(maxsize=args.queue_size)
    pool = Pool(args.num_workers, worker, (input_q, output_q))

    # Get a reference to webcam #0 (the default one)
    # video_capture = cv2.VideoCapture(0)
    video_capture = WebcamVideoStream(src=args.video_source,
                                      width=args.width,
                                      height=args.height).start()
    fps = FPS().start()

    # while video_capture.isOpened():
    while True:
        # Grab a single frame of video
        # ret, frame = video_capture.read()
        myprint("out queue {} and input que size {} video_capture start ".format(output_q.qsize(), input_q.qsize()), time.time())
        frame = video_capture.read()
        myprint("out queue {} and input que size {} ".format(output_q.qsize(), input_q.qsize()), time.time())
        input_q.put(frame)
        myprint("out queue {} and input que size {} ".format(output_q.qsize(), input_q.qsize()), time.time())
        # Only process every other frame of video to save time
        if process_this_frame:
            # COLOR_RGB2BGR
            myprint("out queue {} and input que size {} ".format(output_q.qsize(), input_q.qsize()), time.time())
            cv2.imshow("aa", output_q.get())
            myprint("out queue {} and input que size {} after imshow ".format(output_q.qsize(), input_q.qsize()),
                    time.time())
            # cv2.imshow("aa", frame)
            fps.update()
            # face_process(rgb_small_frame)
            # output_rgb = cv2.cvtColor(output_q.get(), cv2.COLOR_RGB2BGR)
            # t = threading.Thread(target=face_process, name='face_process')  # 线程对象.
            # t.start()  # 启动.
        # process_this_frame = not process_this_frame
        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    fps.stop()
    pool.terminate()

    # Release handle to the webcam
    video_capture.release()
    cv2.destroyAllWindows()
