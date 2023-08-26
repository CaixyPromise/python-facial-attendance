import dlib
import cv2
import time
from utils.putChineseText import cv2AddChineseText
from db.DatabaseHandler import DatabaseHandler
import csv
class FaceRegister:
    def __init__(self, detector, faceId = 1, userName = 'default', interval = 3, faceCount = 3, resize_w = 700,
                 resize_h = 400
                 ):
        # 初始化人脸注册类
        self.faceId = faceId  # 设置人脸ID
        self.userName = userName  # 设置用户名称
        self.interval = interval  # 设置注册人脸的每张间隔时间
        self.faceCount = faceCount  # 设置注册人脸的照片数量
        self.resize_w = resize_w  # 设置画面缩放宽度
        self.resize_h = resize_h  # 设置画面缩放高度

        # 加载关键点检测器和人脸描述符提取器
        self.points_detector = dlib.shape_predictor('./weights/shape_predictor_68_face_landmarks.dat')  # 用于检测68个人脸关键点
        self.face_descriptor_extractor = dlib.face_recognition_model_v1(
            './weights/dlib_face_recognition_resnet_model_v1.dat'
            )  # 用于提取人脸特征描述符
        # 初始化检测器
        self.detector = detector  # 设置使用的人脸检测器，可以是hog、cnn或haar

    def register(self):
        # Initialize counters and timings
        count = 0
        startTime = time.time()
        frameTime = startTime
        show_time = (startTime - 10)
        db = DatabaseHandler()
        user_id = db.add_user(self.userName)

        # Capture video
        cap = cv2.VideoCapture(0)

        while True:
            ret, frame = cap.read()
            frame = cv2.resize(frame, (self.resize_w, self.resize_h))
            frame = cv2.flip(frame, 1)

            # 检测人脸
            face_detections = self.detector.detect_faces(frame)

            for face in face_detections:
                l, t, r, b = self.detector.get_face_boundaries(face)
                face_rect = dlib.rectangle(l, t, r, b)

                # 检测面部特征(68个)
                points = self.points_detector(frame, face_rect)

                # 在图像上绘制面部特征
                for point in points.parts():
                    cv2.circle(frame, (point.x, point.y), 2, (255, 0, 255), 1)
                cv2.rectangle(frame, (l, t), (r, b), (0, 255, 0), 2)

                # 处理注册间隔
                now = time.time()
                if (now - show_time) < 0.5:
                    frame = cv2AddChineseText(frame, "注册成功 {count}/{faceCount}".format(count=(count + 1), faceCount=self.faceCount), (l, b + 30), textColor=(255, 0, 255), textSize=40)

                if count < self.faceCount:
                    if now - startTime > self.interval:
                        face_descriptor = self.face_descriptor_extractor.compute_face_descriptor(frame, points)
                        face_descriptor = [f for f in face_descriptor]

                        # 将人脸描述符添加到 数据库
                        # line = [self.faceId, self.userName, face_descriptor]
                        db.add_feature(user_id, str(face_descriptor))  # 注意：这里把numpy数组转为字符串存入数据库

                        # 更新对应人脸特征上的注册状态
                        frame = cv2AddChineseText(frame, "注册成功 {count}/{faceCount}".format(count=(count + 1), faceCount=self.faceCount), (l, b + 30), textColor=(255, 0, 255), textSize=40)
                        show_time = time.time()

                        # 重置计时和增量计数器
                        startTime = now
                        count += 1
                else:
                    print('人脸注册完毕')
                    cap.release()
                    cv2.destroyAllWindows()
                    return

            # 展示画面
            cv2.imshow('Face Attendance Demo: Register', frame)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
