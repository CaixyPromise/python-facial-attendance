import time
import dlib
import cv2
import numpy as np
from utils.putChineseText import cv2AddChineseText
from db.DatabaseHandler import DatabaseHandler

class FaceRecognizer:
    def __init__(self, detector, threshold = 0.5, resize_w = 700, resize_h = 400):
        # 初始化人脸识别类
        self.detector = detector  # 设置使用的人脸检测器
        self.threshold = threshold  # 设置人脸识别的阈值
        self.points_detector = dlib.shape_predictor('./weights/shape_predictor_68_face_landmarks.dat')  # 加载关键点检测器
        self.face_descriptor_extractor = dlib.face_recognition_model_v1(
            './weights/dlib_face_recognition_resnet_model_v1.dat'
            )  # 加载人脸特征描述符提取器
        self.resize_w = resize_w  # 设置画面缩放宽度
        self.resize_h = resize_h  # 设置画面缩放高度
        self.db = DatabaseHandler()

        # 加载人脸特征值
        self.feature_list, self.label_list, self.name_list = self.get_feature_list(self.db)
        if (self.feature_list is None):
            print('没有注册的人脸特征, 请先注册人脸特征.')
            exit(-1)

    def get_feature_list(self, db):
        # 从CSV文件中加载已注册的人脸特征
        print('加载注册的人脸特征')
        features_from_db = db.get_features()

        feature_list = None  # 初始化特征列表
        label_list = []  # 初始化标签列表
        name_list = []  # 初始化名称列表

        # 遍历从数据库中获取的每一行
        for item in features_from_db:
            user_id, user_name, feature = item  # 分解为用户ID、用户名和特征描述符
            label_list.append(user_id)  # 添加到标签列表
            name_list.append(user_name)  # 添加到名称列表
            face_descriptor = np.asarray(eval(feature), dtype = np.float64)  # 将字符串的特征描述符转换为numpy数组
            face_descriptor = np.reshape(face_descriptor, (1, -1))  # 调整数组的形状

            if feature_list is None:  # 如果特征列表为空
                feature_list = face_descriptor  # 将当前的特征描述符设置为特征列表
            else:  # 否则
                feature_list = np.concatenate((feature_list, face_descriptor), axis = 0)  # 在现有的特征列表上追加特征描述符
        print('加载注册的人脸特征完成')
        return feature_list, label_list, name_list  # 返回特征列表、标签列表和名称列表

    def recognize(self, frame):
        # 在给定的帧上执行人脸识别
        frameTime = time.time()  # 获取当前时间
        face_time_dict = {}  # 初始化人脸时间字典
        face_info_list = []  # 初始化人脸信息列表
        face_img_list = []  # 初始化人脸图像列表
        person_detect = 0  # 初始化检测到的人数为0
        face_count = 0  # 初始化人脸计数为0
        show_time = (frameTime - 10)  # 设置显示时间

        # 检测人脸
        face_detections = self.detector.detect_faces(frame)
        person_detect = len(face_detections)

        for face in face_detections:
            l, t, r, b = self.detector.get_face_boundaries(face)
            face_rect = dlib.rectangle(l, t, r, b)

            # 检测人脸特征
            points = self.points_detector(frame, face_rect)

            # 提取面部描述符
            face_descriptor = self.face_descriptor_extractor.compute_face_descriptor(frame, points)
            face_descriptor = [f for f in face_descriptor]
            face_descriptor = np.asarray(face_descriptor, dtype = np.float64)

            # 计算欧式距离并找到最接近的匹配
            print(self.feature_list)
            distance = np.linalg.norm((face_descriptor - self.feature_list), axis = 1)
            min_index = np.argmin(distance)
            min_distance = distance[min_index]

            predict_name = "Not recog"
            if min_distance < self.threshold:
                predict_id = self.label_list[min_index]
                predict_name = self.name_list[min_index]

                # 处理识别间隔并更新记录
                need_insert = False
                now = time.time()
                if predict_name in face_time_dict:
                    if (now - face_time_dict[predict_name]) > 3:
                        face_time_dict[predict_name] = now
                        need_insert = True
                else:
                    face_time_dict[predict_name] = now
                    need_insert = True

                if (now - show_time) < 1:
                    frame = cv2AddChineseText(frame, "打卡成功", (l, b + 30), textColor = (0, 255, 0), textSize = 40)

                if need_insert:
                    time_local = time.localtime(face_time_dict[predict_name])
                    face_time = time.strftime("%H:%M:%S", time_local)
                    self.db.add_attendance(predict_id, predict_name)
                    face_count += 1

            # 在画面上绘制结果
            cv2.putText(frame, predict_name + " " + str(round(min_distance, 2)), (l, b + 30), cv2.FONT_ITALIC, 0.8,
                        (0, 255, 0), 2
                        )

        # Display the frame
        cv2.imshow('Face Attendance Demo: Recognition', frame)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
