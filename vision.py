from imutils.video import FPS
import imutils
import cv2


class FaceDetector:
    def __init__(self):
        self.faceCascade = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')

    def start(self):
        self.stream = cv2.VideoCapture(0)
        self.fps = FPS()
        self.fps.start()

    def detect(self):
        grabbed, frame = self.stream.read()
        frame = imutils.resize(frame, width=400)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(50, 50),
            flags=cv2.CASCADE_SCALE_IMAGE,
        )

        # Draw a rectangle around the faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        cv2.imshow('Frame', frame)
        cv2.waitKey(1)
        self.fps.update()
        return faces

    def stop(self):
        cv2.destroyAllWindows()
        self.stream.release()
        self.fps.stop()
        print("[INFO] elapsed time: {:.2f}".format(self.fps.elapsed()))
        print("[INFO] approx. FPS: {:.2f}".format(self.fps.fps()))
