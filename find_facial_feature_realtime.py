# !/usr/bin/python3.6
# -*- coding: utf-8 -*-
# @author breeze

from PIL import Image, ImageDraw
import face_recognition
import cv2
import numpy as np
# Create a VideoCapture object
cap = cv2.VideoCapture(0)

# Check if camera opened successfully
if not cap.isOpened():
    print("Unable to read camera feed")
    exit(-1)

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
index = 0
process_this_frame = True
face_locations = []
while True:
    ret, frame = cap.read()
    if not ret:
        continue

    key = cv2.waitKey(1)
    # Press Q on keyboard to stop recordingq
    if key & 0xFF == ord('q'):
        break
    if key & 0xFF == ord('c'):
        index += 1
    source = np.copy(frame)
    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = small_frame[:, :, ::-1]
    # Only process every other frame of video to save time
    if process_this_frame:
        face_locations = face_recognition.face_locations(rgb_small_frame, model="hog")
    process_this_frame = not process_this_frame

    # Display the results
    for (top, right, bottom, left) in face_locations:
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a box around the face
        print("frame shape", frame.shape)
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        print("bouding box ", (left, top), (right, bottom))

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, "face", (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
        # Display the resulting image
    cv2.imshow('Video', frame)

    # part 2 facial features
    face_landmarks_list = face_recognition.face_landmarks(source)  # source
    print("I found {} face(s) in this photograph.".format(len(face_landmarks_list)))

    for face_landmarks in face_landmarks_list:

        # Print the location of each facial feature in this image
        facial_features = [
            'chin',
            'left_eyebrow',
            'right_eyebrow',
            'nose_bridge',
            'nose_tip',
            'left_eye',
            'right_eye',
            'top_lip',
            'bottom_lip'
        ]

        for facial_feature in facial_features:
            print("The {} in this face has the following points: {}"\
                  .format(facial_feature,face_landmarks[facial_feature]))

        # Let's trace out each facial feature in the image with a line!
        pil_image = Image.fromarray(source)  # source or frame
        d = ImageDraw.Draw(pil_image)

        for facial_feature in facial_features:
            d.line(face_landmarks[facial_feature], width=2)
        import numpy as np

        # pil_image.show()
        img = cv2.cvtColor(np.asarray(pil_image), cv2.COLOR_RGB2BGR)
        cv2.imshow("features", img)
        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# When everything done, release the video capture and video write objects
cap.release()

# Closes all the frames
cv2.destroyAllWindows()
