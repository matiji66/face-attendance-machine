# !/usr/bin/python3.6
# -*- coding: utf-8 -*-
# @author breeze

import os

import face_recognition
import numpy as np
import time

# This is a demo of running face recognition on live video from your webcam. It's a little more complicated than the
# other example, but it includes some basic performance tweaks to make things run a lot faster:
#   1. Process each video frame at 1/4 resolution (though still display it at full resolution)
#   2. Only detect faces in every other frame of video.
# PLEASE NOTE: This example requires OpenCV (the `cv2` library) to be installed only to read from your webcam.
# OpenCV is *not* required to use the face_recognition library. It's only required if you want to run this
# specific demo. If you have trouble installing it, try any of the other demos that don't require it instead.
# Get a reference to webcam #0 (the default one)
# Load a sample picture and learn how to recognize it.
# face_recognition.api.batch_face_locations(images, number_of_times_to_upsample=1, batch_size=128)[source]
# face_recognition.api.compare_faces(known_face_encodings, face_encoding_to_check, tolerance=0.6)
# face_recognition.api.face_distance(face_encodings, face_to_compare)[source]
# face_recognition.api.face_encodings(face_image, known_face_locations=None, num_jitters=1)[source]
# face_recognition.api.face_landmarks(face_image, face_locations=None)[source]
# face_recognition.api.face_locations(img, number_of_times_to_upsample=1, model='hog')[source]
# face_recognition.api.load_image_file(file, mode='RGB')[source]

# save im binary format
# np.array(known_face_encodings).tofile("./dataset/known_face_encodings.bin")
# np.array(known_face_names).tofile("./dataset/known_face_names.bin")
# save im numpy format https://www.cnblogs.com/dmir/p/5009075.html
# np.save(KNOWN_FACE_ENCODINGS, known_face_encodings)
# np.save(KNOWN_FACE_NANE, known_face_names)

data_path = "./dataset"  # 相关文件保存路径
KNOWN_FACE_ENCODINGS = "./dataset/known_face_encodings.npy"  # 已知人脸向量
KNOWN_FACE_NANE = "./dataset/known_face_name.npy"  # 已知人脸名称

new_images = []  # 同时将最新的文件记录
processd_images = []  # 将已经处理的图片进行保存,在下次启动的时候跳过在该文件中已经出现过的文件,并在程序结束之前与new_images 进行合并然后保存,并覆盖该文件
known_face_names = []  # 已知人脸名称
known_face_encodings = []  # 已知人脸编码
name_and_encoding = "./dataset/face_encodings.txt"

image_thread = 0.15
# TODO  加入多线程,或者是线程池,对每个人的文件采用多线程的处理


def encoding_images(path):
    """
    对path路径下的子文件夹中的图片进行编码,
    TODO:
        对人脸数据进行历史库中的人脸向量进行欧式距离的比较,当距离小于某个阈值的时候提醒:
        如果相似的是本人,则跳过该条记录,并提醒已经存在,否则警告人脸过度相似问题,
    :param path:
    :return:
    """
    with open(name_and_encoding, 'w') as f:
        subdirs = [os.path.join(path, x) for x in os.listdir(path) if os.path.isdir(os.path.join(path, x))]
        for subdir in subdirs:
            print('process image name :', subdir)
            person_image_encoding = []
            for y in os.listdir(subdir):
                print("image name is ", y)
                _image = face_recognition.load_image_file(os.path.join(subdir, y))
                face_encodings = face_recognition.face_encodings(_image)
                name = os.path.split(subdir)[-1]
                if face_encodings and len(face_encodings) == 1:
                    if len(person_image_encoding) == 0:
                        person_image_encoding.append(face_encodings[0])
                        known_face_names.append(name)
                        continue
                    for i in range(len(person_image_encoding)):
                        distances = face_recognition.compare_faces(person_image_encoding, face_encodings[0], tolerance=image_thread)
                        if False in distances:
                            person_image_encoding.append(face_encodings[0])
                            known_face_names.append(name)
                            print(name, " new feature")
                            f.write(name + ":" + str(face_encodings[0]) + "\n")
                            break
                    # face_encoding = face_recognition.face_encodings(_image)[0]
                    # face_recognition.compare_faces()
            known_face_encodings.extend(person_image_encoding)
            bb = np.array(known_face_encodings)
            print("--------")
    np.save(KNOWN_FACE_ENCODINGS, known_face_encodings)
    np.save(KNOWN_FACE_NANE, known_face_names)


def encoding_images_mult_thread(path,threads=8):
    """
    对path路径下的子文件夹中的图片进行编码,
    TODO:
        对人脸数据进行历史库中的人脸向量进行欧式距离的比较,当距离小于某个阈值的时候提醒:
        如果相似的是本人,则跳过该条记录,并提醒已经存在,否则警告人脸过度相似问题,

    :param path:
    :return:
    """
    # with open("./dataset/encoded_face_names.txt", 'w') as f:
    #     lines = f.readlines()
    #     print(lines)

    with open(name_and_encoding, 'w') as f:
        subdirs = [os.path.join(path, x) for x in os.listdir(path) if os.path.isdir(os.path.join(path, x))]
        # if
        for subdir in subdirs:
            print('---name :', subdir)
            person_image_encoding = []
            for y in os.listdir(subdir):
                print("image name is ", y)
                _image = face_recognition.load_image_file(os.path.join(subdir, y))
                face_encodings = face_recognition.face_encodings(_image)
                name = os.path.split(subdir)[-1]
                if face_encodings and len(face_encodings) == 1:
                    if len(person_image_encoding) == 0:
                        person_image_encoding.append(face_encodings[0])
                        known_face_names.append(name)
                        continue

                    for i in range(len(person_image_encoding)):
                        distances = face_recognition.compare_faces(person_image_encoding, face_encodings[0], tolerance=image_thread)
                        if False in distances:
                            person_image_encoding.append(face_encodings[0])
                            known_face_names.append(name)
                            print(name, " new feature")
                            f.write(name + ":" + str(face_encodings[0]) + "\n")
                            break

                    # face_encoding = face_recognition.face_encodings(_image)[0]
                    # face_recognition.compare_faces()
            known_face_encodings.extend(person_image_encoding)
            bb = np.array(known_face_encodings)
            print("--------")

    np.save(KNOWN_FACE_ENCODINGS, known_face_encodings)
    np.save(KNOWN_FACE_NANE, known_face_names)


def encoding_ones_images(name):
    """
    对path路径下的子文件夹中的图片进行编码,
    TODO:
        对人脸数据进行历史库中的人脸向量进行欧式距离的比较,当距离小于某个阈值的时候提醒:
        如果相似的是本人,则跳过该条记录,并提醒已经存在,否则警告人脸过度相似问题,

    :param path:
    :return:
    """
    # with open("./dataset/encoded_face_names.txt", 'w') as f:
    #     lines = f.readlines()
    #     print(lines)

    with open(name_and_encoding, 'w') as f:
        image_dirs = os.path.join(data_path, name)
        files = [os.path.join(image_dirs, x) for x in os.listdir(image_dirs) if os.path.isfile(os.path.join(image_dirs, x))]
        print('---name :', files)
        person_image_encoding = []
        for image_path in files:
            print("image name is ", image_path)
            _image = face_recognition.load_image_file(image_path )
            face_encodings = face_recognition.face_encodings(_image)
            # name = os.path.split(image_path)[1]
            if face_encodings and len(face_encodings) == 1:
                if len(person_image_encoding) == 0:
                    person_image_encoding.append(face_encodings[0])
                    known_face_names.append(name)
                    continue

                for i in range(len(person_image_encoding)):
                    distances = face_recognition.compare_faces(person_image_encoding, face_encodings[0], tolerance=image_thread)
                    if False in distances:
                        person_image_encoding.append(face_encodings[0])
                        known_face_names.append(name)
                        print(name, " new feature")
                        f.write(name + ":" + str(face_encodings[0]) + "\n")
                        break

                # face_encoding = face_recognition.face_encodings(_image)[0]
                # face_recognition.compare_faces()
        known_face_encodings.extend(person_image_encoding)
        bb = np.array(known_face_encodings)
        print("--------")

    KNOWN_FACE_ENCODINGS = "./dataset/known_face_encodings_{}.npy"  # 已知人脸向量
    KNOWN_FACE_NANE = "./dataset/known_face_name_{}.npy"  # 已知人脸名称
    np.save(KNOWN_FACE_ENCODINGS.format(int(time.time())), known_face_encodings)
    np.save(KNOWN_FACE_NANE.format(int(time.time())), known_face_names)


def load_encodings():
    """
    加载保存的历史人脸向量,以及name向量,并返回
    :return:
    """
    known_face_encodings = np.load(KNOWN_FACE_ENCODINGS)
    known_face_names = np.load(KNOWN_FACE_NANE)
    if not os.path.exists(KNOWN_FACE_NANE) or not os.path.exists(KNOWN_FACE_ENCODINGS):
        encoding_images(data_path)
    aa = [file for file in os.listdir(data_path) if os.path.isfile(os.path.join(data_path, file)) and file.endswith("npy")]
    # ("known_face_encodings_") or file.startswith("known_face_name_"))
    for data in aa:
        if data.startswith('known_face_encodings_'):
            tmp_face_encodings = np.load(os.path.join(data_path,data))
            known_face_encodings = np.concatenate((known_face_encodings, tmp_face_encodings), axis=0)
            print("load ", data)
        elif data.startswith('known_face_name_'):
            tmp_face_name = np.load(os.path.join(data_path, data))
            known_face_names = np.concatenate((known_face_names, tmp_face_name), axis=0)
            print("load ", data)
        else:
            print('skip to load original ', data)
    return known_face_encodings,known_face_names


def test_load():
    face_encodings, face_names = load_encodings()
    print("===========face_encodings================")
    print(face_encodings)
    print("===========================")
    print(face_names)
    print("===========face_names================")


if __name__ == '__main__':
    # encoding_images()
    # encoding_ones_images('jim')
    # try:
    #     encoding_images(data_path)  # encoding all images in data_path sub dir
    # except Exception as e:
    #     print("ERROR : create image encoding failed ! ")

    # 测试加载数据库
    test_load()
