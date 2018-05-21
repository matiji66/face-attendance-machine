import cv2

# This is a demo of running face recognition on a video file and saving the results to a new video file.
#
# PLEASE NOTE: This example requires OpenCV (the `cv2` library) to be installed only to read from your webcam.
# OpenCV is *not* required to use the face_recognition library. It's only required if you want to run this
# specific demo. If you have trouble installing it, try any of the other demos that don't require it instead.

# Open the input movie file
# input_movie = cv2.VideoCapture("outpy1525941951.7225914.avi")
# length = int(input_movie.get(cv2.CAP_PROP_FRAME_COUNT))

# Create an output movie file (make sure resolution/frame rate matches input video!)
# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# output_movie = cv2.VideoWriter('output.avi', fourcc, 29.97, (640, 360))

import os

files = [path for path in os.listdir("./videos") if os.path.isfile(path) and path.endswith(".avi")]

frame_number = 0
for avi in files:
    input_movie = cv2.VideoCapture(avi)
    length = int(input_movie.get(cv2.CAP_PROP_FRAME_COUNT))
    while True:
        # Grab a single frame of video
        ret, frame = input_movie.read()
        frame_number += 1

        # Quit when the input video file ends
        if not ret:
            break
        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_frame = frame[:, :, ::-1]
        cv2.imwrite("images/image_{}.jpg".format(frame_number), frame)
    # All done!
    input_movie.release()

