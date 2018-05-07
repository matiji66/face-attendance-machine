# face-card-machine
借助 opencv 以及开源的的dlib做一个python版的人脸打卡机器.可以进行人脸识别确认,并记录识别出来的人脸结果.



## first run capture.py to capture image 采集人脸照片
将采集的照片,根据name保存到./dataset路径下.




## second run encoding_images.py to encoding faces image 将人脸照片保存到npy中
对人脸编码进行持久化,避免每次重新开始编码,提升程序启动速度




## third run facerec_from_webcam_faster.py for face recognize and persist to local file(or mysql)
进行人脸识别,身份确认,并每隔一段时间将识别结果持久化到 ./dataset/face_record.txt中,用来历史查询.

至此一个简单的人脸打卡工程就结束了


为了兼顾准确率,以及识别精度,采集人脸多个角度的照片(8张),并降低dlib的人脸比对的阈值,提升准确路,避免人脸识别错误.



