from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import ZeroPadding2D
from keras import backend as K
from os import listdir
import numpy as np
import cv2
from keras.preprocessing.image import img_to_array
from keras.models import load_model


model= load_model("multiclass.h5")
def loadImages(path):
    # return array of images

    imagesList = listdir(path)
    loadedImages = []
    for image in imagesList:
        img =cv2.imread(path + image)
        img = cv2.resize(img, (64, 64))
        image = img_to_array(img)
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        loadedImages.append(image)

    return loadedImages

path = "SingleImg/"

# your images in an array
imgs = loadImages(path)

i=1
for img in imgs:
    value=model.predict(img)
    classname = value.argmax(axis=-1)
    print(i," ",value)
    i+=1