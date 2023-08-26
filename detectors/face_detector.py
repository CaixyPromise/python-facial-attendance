import dlib
import cv2

class FaceDetector:
    def __init__(self, detector_type = 'haar'):
        # 初始化并加载人脸检测器
        self.hog_face_detector = dlib.get_frontal_face_detector()  # 使用HOG特征的检测器
        self.cnn_detector = dlib.cnn_face_detection_model_v1('weights/mmod_human_face_detector.dat')  # 使用CNN的检测器
        self.haar_face_detector = cv2.CascadeClassifier('weights/haarcascade_frontalface_default.xml')  # 使用Haar特征的检测器
        self.detector_type = detector_type  # 设置当前使用的检测器类型

    def detect_faces(self, frame):
        # 根据所选的检测器类型进行人脸检测
        if self.detector_type == 'hog':
            return self.hog_face_detector(frame, 1)  # 使用HOG检测器
        elif self.detector_type == 'cnn':
            return self.cnn_detector(frame, 1)  # 使用CNN检测器
        elif self.detector_type == 'haar':
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # 将图像转换为灰度格式
            return self.haar_face_detector.detectMultiScale(frame_gray, minNeighbors = 7, minSize = (100, 100))  # 使用Haar检测器
        else:
            raise ValueError("Invalid detector type")  # 如果传入的检测器类型无效，则抛出错误

    def get_face_boundaries(self, face):
        # 获取人脸的边界坐标
        l, t, r, b = None, None, None, None
        if self.detector_type == 'hog':
            l, t, r, b = face.left(), face.top(), face.right(), face.bottom()  # 对于HOG和CNN，使用dlib的rectangle方法
        elif self.detector_type == 'cnn':
            l, t, r, b = face.rect.left(), face.rect.top(), face.rect.right(), face.rect.bottom()
        elif self.detector_type == 'haar':
            l, t, r, b = face[0], face[1], face[0] + face[2], face[1] + face[3]  # 对于Haar，直接从检测结果中提取坐标

        nonnegative = lambda x: x if x >= 0 else 0  # 确保坐标是非负的
        return map(nonnegative, (l, t, r, b))
