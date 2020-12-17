import numpy as np
import pandas as pd
import os
import argparse
import errno
import scipy.misc
import dlib
import cv2

from skimage.feature import hog

# initialization
image_height = 48
image_width = 48
window_size = 24
window_step = 6
ONE_HOT_ENCODING = True
GET_LANDMARKS = False
GET_HOG_FEATURES = False
GET_HOG_IMAGES = False
GET_HOG_WINDOWS_FEATURES = False
SELECTED_LABELS = []
IMAGES_PER_LABEL = 500
OUTPUT_FOLDER_NAME = "./all_features"
emotion_id = {'angry':0, 'disgust':1, 'fear':2, 'happy':3, 'sad':4, 'surprise':5, 'neutral':6}

# parse arguments and initialize variables:
parser = argparse.ArgumentParser()
parser.add_argument("-j", "--jpg", default="no", help="save images as .jpg files")
parser.add_argument("-l", "--landmarks", default="yes", help="extract Dlib Face landmarks")
parser.add_argument("-ho", "--hog", default="yes", help="extract HOG features")
parser.add_argument("-hw", "--hog_windows", default="yes", help="extract HOG features from a sliding window")
parser.add_argument("-hi", "--hog_images", default="no", help="extract HOG images")
parser.add_argument("-o", "--onehot", default="yes", help="one hot encoding")
parser.add_argument("-e", "--expressions", default="0,1,2,3,4,5,6", help="choose the faciale expression you want to use: 0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral")
args = parser.parse_args()
if args.jpg == "yes":
    SAVE_IMAGES = True
if args.landmarks == "yes":
    GET_LANDMARKS = True
if args.hog == "yes":
    GET_HOG_FEATURES = True
if args.hog_windows == "yes":
    GET_HOG_WINDOWS_FEATURES = True
if args.hog_images == "yes":
    GET_HOG_IMAGES = True
if args.onehot == "yes":
    ONE_HOT_ENCODING = True
if args.expressions != "":
    expressions  = args.expressions.split(",")
    for i in range(0,len(expressions)):
        label = int(expressions[i])
        if (label >=0 and label<=6 ):
            SELECTED_LABELS.append(label)
if SELECTED_LABELS == []:
    SELECTED_LABELS = [0,1,2,3,4,5,6]
print( str(len(SELECTED_LABELS)) + " expressions")

# loading Dlib predictor and preparing arrays:
print( "preparing")
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
original_labels = [0, 1, 2, 3, 4, 5, 6]
new_labels = list(set(original_labels) & set(SELECTED_LABELS))
nb_images_per_label = list(np.zeros(len(new_labels), 'uint8'))
try:
    os.makedirs(OUTPUT_FOLDER_NAME)
except OSError as e:
    if e.errno == errno.EEXIST and os.path.isdir(OUTPUT_FOLDER_NAME):
        pass
    else:
        raise

def get_landmarks(image, rects):
    # this function have been copied from http://bit.ly/2cj7Fpq
    if len(rects) > 1:
        raise BaseException("TooManyFaces")
    if len(rects) == 0:
        raise BaseException("NoFaces")
    return np.matrix([[p.x, p.y] for p in predictor(image, rects[0]).parts()])

def get_new_label(label, one_hot_encoding=False):
    if one_hot_encoding:
        new_label = new_labels.index(label)
        label = list(np.zeros(len(new_labels), 'uint8'))
        label[new_label] = 1
        return label
    else:
        return new_labels.index(label)

def sliding_hog_windows(image):
    hog_windows = []
    for y in range(0, image_height, window_step):
        for x in range(0, image_width, window_step):
            window = image[y:y+window_size, x:x+window_size]
            hog_windows.extend(hog(window, orientations=8, pixels_per_cell=(8, 8),
                                            cells_per_block=(1, 1), visualize=False))
    return hog_windows

for usage in ["train", "validation"]:
    if not os.path.exists(os.path.join(OUTPUT_FOLDER_NAME, usage)):
        try:
            os.makedirs(OUTPUT_FOLDER_NAME + '/' + usage)
        except OSError as e:
            if e.errno == errno.EEXIST and os.path.isdir(OUTPUT_FOLDER_NAME):
               pass
            else:
                raise
    print( "convering set: " + usage + "...")
    images = []
    labels_list = []
    landmarks = []
    hog_features = []
    hog_images = []
    for emotion in os.listdir(usage):
        print("NOTE | addressing " + emotion + "...")
        for img in os.listdir(os.path.join(usage, emotion)):
            # print("INFO | img:", img)
            image = cv2.imread(os.path.join(usage, emotion, img), 0)
            images.append(image)
            if GET_HOG_WINDOWS_FEATURES:
                features = sliding_hog_windows(image)
                f, hog_image = hog(image, orientations=8, pixels_per_cell=(16, 16),
                                        cells_per_block=(1, 1), visualize=True)
                hog_features.append(features)
                if GET_HOG_IMAGES:
                    hog_images.append(hog_image)
            elif GET_HOG_FEATURES:
                features, hog_image = hog(image, orientations=8, pixels_per_cell=(16, 16),
                                        cells_per_block=(1, 1), visualize=True)
                hog_features.append(features)
                if GET_HOG_IMAGES:
                    hog_images.append(hog_image)
            if GET_LANDMARKS:
                scipy.misc.imsave('temp.jpg', image)
                image2 = cv2.imread('temp.jpg', 0)
                face_rects = [dlib.rectangle(left=1, top=1, right=47, bottom=47)]
                face_landmarks = get_landmarks(image2, face_rects)
                landmarks.append(face_landmarks)            
            labels_list.append(get_new_label(emotion_id[emotion], one_hot_encoding=ONE_HOT_ENCODING))
            nb_images_per_label[get_new_label(emotion_id[emotion])] += 1

    print("INFO | image.shape", np.array(images).shape)
    print("INFO | labels.shape", np.array(labels_list).shape)
    print("INFO | landmarks.shape", np.array(landmarks).shape)
    print("INFO | hog_features.shape", np.array(hog_features).shape)
    np.save(OUTPUT_FOLDER_NAME + '/' + usage + '/images.npy', images)
    if ONE_HOT_ENCODING:
        np.save(OUTPUT_FOLDER_NAME + '/' + usage + '/labels.npy', labels_list)
    else:
        np.save(OUTPUT_FOLDER_NAME + '/' + usage + '/labels.npy', labels_list)
    if GET_LANDMARKS:
        np.save(OUTPUT_FOLDER_NAME + '/' + usage + '/landmarks.npy', landmarks)
    if GET_HOG_FEATURES or GET_HOG_WINDOWS_FEATURES:
        np.save(OUTPUT_FOLDER_NAME + '/' + usage + '/hog_features.npy', hog_features)
        if GET_HOG_IMAGES:
            np.save(OUTPUT_FOLDER_NAME + '/' + usage + '/hog_images.npy', hog_images)
