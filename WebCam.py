import cv2
from keras.models import load_model
from keras.preprocessing.image import img_to_array
import numpy as np
import dlib


class EmoDetection():
    def __init__(self):
        # detection_model_path = 'cv2/haarcascade_frontalface_default.xml'
        self.emotion_model_path = 'models/_mini_XCEPTION.102-0.66.hdf5'

        # Define face detector
        # face_detector = cv2.CascadeClassifier(detection_model_path)
        self.dlib_face_detector = dlib.get_frontal_face_detector()

        # Define emotion classifier
        self.emotion_classifier = load_model(self.emotion_model_path, compile=False)
        self.emotions = ["angry", "disgust", "scared", "happy", "sad", "surprised", "neutral"]
        self.camera = cv2.VideoCapture(0)
        self.faceDetection()


    def faceDetection(self):
        while True:
            # read through camera capture
            _, frame = self.camera.read()
            # read through image
            #frame = cv2.imread('emotions/kc.jpg')
            flip_frame = cv2.flip(frame, 1)
            gray_img = cv2.cvtColor(flip_frame, cv2.COLOR_BGR2GRAY)
            # cv_faces_detected= face_detector.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=6, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
            dlib_faces_detected = self.dlib_face_detector(gray_img)

            # #haar detect
            # for (x,y,w,h) in cv_faces_detected:
            #     # faces = sorted(faces, reverse=True,
            #     #                key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
            #     # (fX, fY, fW, fH) = faces
            #
            #     cv2.rectangle(flip_frame, (x, y), (x + w, y + h),(255, 0, 0), 2)
            #
            #     # Extract the ROI of the face from the grayscale image, resize it to a fixed 28x28 pixels, and then prepare
            #     # the ROI for classification via the CNN
            #     roi = gray_img[y:y + h, x:x + w]
            #     roi = cv2.resize(roi, (64, 64))
            #     roi = roi.astype("float") / 255.0
            #     roi = img_to_array(roi)
            #     roi = np.expand_dims(roi, axis=0)
            #
            #     #classify the emotion and gives the prediction
            #     preds = emotion_classifier.predict(roi)[0]
            #     #emotion_probability = np.max(preds)
            #     label = emotions[preds.argmax()]
            #     cv2.putText(flip_frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            if len(dlib_faces_detected) < 1:
                cv2.putText(flip_frame, "No face detected", (180, 420), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # dlib detect
            for faces in dlib_faces_detected:
                x1 = faces.left()
                y1 = faces.top()
                x2 = faces.right()
                y2 = faces.bottom()

                cv2.rectangle(flip_frame, (x1, y1), (x2, y2), (255, 255, 255), 2)

                # Extract the ROI of the face from the grayscale image, resize it to a fixed 64x64 pixels, and then prepare
                # the ROI for classification via the CNN
                try:
                    roi = gray_img[y1:y2, x1:x2]
                    roi = cv2.resize(roi, (64, 64))
                    roi = roi.astype("float") / 255.0
                    roi = img_to_array(roi)
                    roi = np.expand_dims(roi, axis=0)

                    # classify the emotion and gives the prediction
                    preds = self.emotion_classifier.predict(roi)[0]
                    # emotion_probability = np.max(preds)
                    label = self.emotions[preds.argmax()]
                except:
                    pass

                cv2.putText(flip_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

            cv2.putText(flip_frame, "Esc: Quit", (20, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
            resized_img = cv2.resize(flip_frame, (1000, 700))
            cv2.imshow('Face Expression Recognition', resized_img)


            # Press esc to escape
            k = cv2.waitKey(30) & 0xff
            if k == 27:
                break

        self.camera.release()
        cv2.destroyAllWindows()




