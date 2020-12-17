import numpy as np
import pandas as pd
import os
import cv2

emotion_id = {'angry':0, 'disgust':1, 'fear':2, 'happy':3, 'sad':4, 'surprise':5, 'neutral':6}

for usage in ["validation"]:
    print( "convering set: " + usage + "...")
    with open(usage + ".csv", "w") as csv_file:
        print("emotions,pixels,usage", file = csv_file)
        for emotion in os.listdir(usage):
            print("NOTE | addressing " + emotion + "...")
            for img in os.listdir(os.path.join(usage, emotion)):
                # print("INFO | img:", img)
                image = cv2.imread(os.path.join(usage, emotion, img))
                print("%d,%s,%s" % (emotion_id[emotion], img, usage), file=csv_file)
                