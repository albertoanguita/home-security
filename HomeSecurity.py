import logging
import os
import pickle

import cv2
import numpy as np
from jacpy.geometry import geoUtils
from mtcnn import mtcnn

import HumansModel
import TimedSample
from FaceEncoder import FaceEncoder

# The HomeSecurity system is designed to video monitor the entrance of a home and detect if an intruder (not
# registered in the system) is pictured. To do so, the system uses an attached web-cam and two different NNs.
# - HumanDetector: detects if there is a human in the frame. This is a relatively lightweight calculation, so it
# can be run continuously
# - FaceRecognizer: searches for faces in the frame and tries to match them against a set of registered faces. If
# the face is registered, there everything is ok. If there is no face found, or the found face does not match
# against any the registered ones, then and intrusion alarm is fired off, triggering the necessary notifications.


timedStore = TimedSample.TimedSampleStore(5000)

encodings_path = 'encodings/encodings.pkl'
encodings_weights = 'facenet_keras_weights.h5'

required_size = (160, 160)
human_threshold = 0.0
face_threshold = 0.99
recognition_threshold = 0.20


def init():
    logging.info("Starting system")

    humanModel = loadHumanMode()
    face_detector = mtcnn.MTCNN()
    face_encoder = FaceEncoder(face_detector, 'facenet_keras_weights.h5', 'encodings/encodings.pkl', 'Faces2/')
    if not os.path.isfile('encodings/encodings.pkl') or face_encoder.HasNewSamples():
        face_encoder.GenerateEncodings()

    face_encoder.LoadEncodings()

    mainLoop(humanModel, face_encoder)

def loadHumanMode():
    humanModel = HumansModel.HumansModel()
    humanModel.load_weights('weights/detect_humans_weight_IV2.h5')
    logging.info("Human model loaded correctly")
    return humanModel

def load_pickle(path):
    with open(path, 'rb') as f:
        encoding_dict = pickle.load(f)
    return encoding_dict

def divide(frame: cv2.typing.MatLike):
    shape = frame.shape
    rows = shape[0]
    cols = shape[1]
    partitions = geoUtils.partition2DGeometry(cols, rows, 320, 320, 0.2, growRatio=1.5)
    return partitions

def detectHumans(img: cv2.typing.MatLike, partitions, humanModel):
    # humanParts = []
    # noHumanParts = []
    # partsWithColor = []
    partitionedImages = list(map(lambda p : img[p[0][1]:p[1][1], p[0][0]:p[1][0]], partitions))
    processedImages = np.array(list(map(lambda i : cv2.resize(i, required_size), partitionedImages)))
    humanPredictions = humanModel.predict(processedImages)
    humanPrediction = max(humanPredictions)

    return humanPrediction > human_threshold


def detectFaces(frame, img, encoder):
    known, unknown = encoder.DetectFaces(img, face_threshold, recognition_threshold)

    for aKnown in known:
        name = aKnown[0]
        distance = aKnown[1]
        pt_1 = aKnown[2]
        pt_2 = aKnown[3]
        cv2.rectangle(frame, pt_1, pt_2, (0, 255, 0), 2)
        cv2.putText(frame, name + f'__{distance:.2f}', (pt_1[0], pt_1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 200, 200), 2)

    for anUnknown in unknown:
        pt_1 = anUnknown[0]
        pt_2 = anUnknown[1]
        cv2.rectangle(frame, pt_1, pt_2, (0, 0, 255), 2)
        cv2.putText(frame, 'unknown', pt_1, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
    return frame

def get_face(img, box):
    x1, y1, width, height = box
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height
    face = img[y1:y2, x1:x2]
    return face, (x1, y1), (x2, y2)


def addFps(frame, timedStore):
    fps = timedStore.count() / 5.0
    cv2.putText(frame, str(fps), (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 1,
                (255, 0, 0), 2)
    return frame


def mainLoop(humanModel, face_encoder):
    logging.info("Initiating main loop")
    cap = cv2.VideoCapture(0)
    logging.info("Video capture initiated")
    frame_count = 0

    while cap.isOpened():
        success, frame = cap.read()

        if not success:
            logging.error("Camera is not accessible. Exiting program!!!")
            break

        logging.debug("Processing a new frame")
        timedStore.add(True)

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        partitions = divide(frame)

        if (detectHumans(img_rgb, partitions, humanModel)):
            logging.warning(f"Human detected in frame {frame_count}")
            # frame = detect.detect(frame, face_detector, face_encoder, encoding_dict)
            frame = detectFaces(frame, img_rgb, face_encoder)
        else:
            logging.debug(f'No human detected in frame {frame_count}')

        frame = addFps(frame, timedStore)

        cv2.imshow('camera', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_count += 1



init()