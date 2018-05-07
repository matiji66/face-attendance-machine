# !/usr/bin/python3.6
# -*- coding: utf-8 -*-
# @author breeze

import time

import cv2
import face_recognition
import win32com.client

import encoding_images

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


# Get a reference to webcam #0 (the default one)
video_capture = cv2.VideoCapture(0)

# 语音模块
speaker = win32com.client.Dispatch("SAPI.SpVoice")

name = "Unknown"
current_names = []
last_time = time.time()
known_face_names = []
known_face_encodings = []
known_face_encodings, known_face_names = encoding_images.load_encodings()

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True
TIME_DIFF = 20  # 演示的时候用,打卡可以设置为86400
name_record = "./dataset/face_record.txt"


# 处理每一条识别的记录
def process_face_records(name):
    global current_names, last_time
    print("global current_names {}, last_time {}".format( current_names, last_time))

    # 判断是不是在识别的列表中,不在的话就进行问候
    if name not in current_names:
        print("ts ====", last_time, time.time())
        current_names.append(name)
        print("Hello {}, nice to meet you! ".format(name))
        speaker.Speak("Hello {}, nice to meet you! ".format(name))
    # 在一定时间内,清空已经识别的人, 并进行
    if last_time < time.time() - TIME_DIFF:  # 每隔一段时间清空一下检测到的人
        last_time = time.time()
        time_format = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        print(time_format + " update last_time and clear current_names.")
        with open(name_record, 'a') as f:
            if len(current_names) > 0:
                f.writelines("{}:{} \n".format(time_format, str(current_names)))
        # print("======", current_names)
        current_names.clear()


while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()

    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = small_frame[:, :, ::-1]

    # Only process every other frame of video to save time
    if process_this_frame:
        # Find all the faces and face encodings in the current frame of video
        # face_locations = face_recognition.face_locations(rgb_small_frame, model="cnn")
        face_locations = face_recognition.face_locations(rgb_small_frame, model="hog")
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)

            ts1 = (int(round(time.time() * 1000)))
            matches = face_recognition.compare_faces( \
                known_face_encodings=known_face_encodings, face_encoding_to_check=face_encoding, tolerance=0.3)
            ts2 = (int(round(time.time() * 1000)))
            print("matches ts1 {} ts2{} cost {}ms".format(ts1, ts2, ts2 - ts1))  # matches
            # If a match was found in known_face_encodings, just use the first one.
            if True in matches:
                first_match_index = matches.index(True)
                name = known_face_names[first_match_index]
            face_names.append(name)
    process_this_frame = not process_this_frame

    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4
        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
        print("process_face_records====")
        # say hello
        process_face_records(name)

    # Display the resulting image
    cv2.imshow('Video', frame)
    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()
