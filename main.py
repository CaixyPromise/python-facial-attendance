import cv2
from face_recognition.recognition.face_recognizer import FaceRecognizer
from face_recognition.registration.face_register import FaceRegister
from face_recognition.detectors.face_detector import FaceDetector

class FaceAttendanceSystem:
    def __init__(self, mode = 'reg',
                       detector = 'haar',
                       faceId = 1,
                       userName = 'default',
                       interval = 3,
                       faceCount = 3,
                       threshold = 0.5):
        self.mode = mode
        self.detector = FaceDetector(detector_type = detector)
        self.register = FaceRegister(detector = detector, faceId = faceId, userName = userName, interval = interval, faceCount = faceCount)
        self.recognizer = FaceRecognizer(detector = detector, threshold = threshold)

    def register(self):
        self.register.register()

    def recognition(self, video_src):
        cap = cv2.VideoCapture(video_src)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            self.recognizer.recognize(frame)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()


    def run(self, video_src = 0):
        if self.mode == 'reg':
            self.register.register()
        elif self.mode == 'recog':
            self.recognition(video_src)
        else:
            raise ValueError("Invalid mode")

# 创建系统实例并运行
system = FaceAttendanceSystem(mode='reg', detector='haar', faceId=1, userName='John')
system.run()

