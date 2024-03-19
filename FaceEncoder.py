import os
import pickle

import cv2
from jacpy.hash import hashUtils
from mtcnn import mtcnn, MTCNN
import numpy as np
from scipy.spatial.distance import cosine
from sklearn.preprocessing import Normalizer

from architecture import InceptionResNetV2


required_shape = (160,160)
l2_normalizer = Normalizer('l2')

def normalize(img):
    mean, std = img.mean(), img.std()
    return (img - mean) / std

def load_pickle(path):
    with open(path, 'rb') as f:
        encoding_dict = pickle.load(f)
    return encoding_dict

def get_face(img, box):
    x1, y1, width, height = box
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height
    face = img[y1:y2, x1:x2]
    return face, (x1, y1), (x2, y2)

def get_encode(face_encoder, face, size):
    face = normalize(face)
    face = cv2.resize(face, size)
    encode = face_encoder.predict(np.expand_dims(face, axis=0))[0]
    return encode

class FaceEncoder:

    def __init__(self, face_detector: MTCNN, weights_path: str, encodings_path: str, samples_dir: str):
        self.__face_detector = face_detector
        self.__face_encoder = InceptionResNetV2()
        self.__face_encoder.load_weights(weights_path)
        self.__encodings_path = encodings_path
        self.__samples_dir = samples_dir
        self.__encoding_dict = dict()


    def HasNewSamples(self):
        hashPath = self.__samples_dir + 'hash.txt'
        hash = hashUtils.digestHexFilesInPath(self.__samples_dir, 1)

        if os.path.isfile(hashPath):
            with open(hashPath, 'r') as hashFile:
                lines = hashFile.readlines()
                if len(lines) == 1 and lines[0] == hash:
                    return False

        return True



    def GenerateEncodings(self):
        face_detector = mtcnn.MTCNN()
        for face_names in os.listdir(self.__samples_dir):
            person_dir = os.path.join(self.__samples_dir, face_names)
            if not os.path.isdir(person_dir):
                continue
            encodes = []
            for image_name in os.listdir(person_dir):
                image_path = os.path.join(person_dir, image_name)

                img_BGR = cv2.imread(image_path)
                img_RGB = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2RGB)

                x = face_detector.detect_faces(img_RGB)
                x1, y1, width, height = x[0]['box']
                x1, y1 = abs(x1), abs(y1)
                x2, y2 = x1 + width, y1 + height
                face = img_RGB[y1:y2, x1:x2]

                face = normalize(face)
                face = cv2.resize(face, required_shape)
                face_d = np.expand_dims(face, axis=0)
                encode = self.__face_encoder.predict(face_d)[0]
                encodes.append(encode)

            if encodes:
                encode = np.sum(encodes, axis=0)
                encode = l2_normalizer.transform(np.expand_dims(encode, axis=0))[0]
                self.__encoding_dict[face_names] = encode

        with open(self.__encodings_path, 'wb') as file:
            pickle.dump(self.__encoding_dict, file)


    def LoadEncodings(self):
        self.__encoding_dict = load_pickle(self.__encodings_path)
        return

    def DetectFaces(self, img, face_threshold, recognition_threshold):
        unknown = []
        known = []
        faces = self.__face_detector.detect_faces(img)
        for face in faces:
            if face['confidence'] < face_threshold:
                continue
            face, pt_1, pt_2 = get_face(img, face['box'])
            encode = get_encode(self.__face_encoder, face, required_shape)
            encode = l2_normalizer.transform(encode.reshape(1, -1))[0]
            name = 'unknown'

            distance = float("inf")
            for db_name, db_encode in self.__encoding_dict.items():
                dist = cosine(db_encode, encode)
                if dist < recognition_threshold and dist < distance:
                    name = db_name
                    distance = dist

            if name != 'unknown':
                known.append((name, distance, pt_1, pt_2))
            else:
                unknown.append((pt_1, pt_2))

        return known, unknown



# def LoadFaceEncodings(face_encoder, path):
