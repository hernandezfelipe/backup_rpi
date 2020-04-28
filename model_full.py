import os
import numpy as np
from keras.models import model_from_json
from keras.optimizers import Adam
import cv2
from time import time
import sys
import psutil
from collections import Counter
from scipy import stats
import matplotlib.pyplot as plt
# from letras.model_plate import read_plate
from ncs_inference import predict
from random import random

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# p = psutil.Process(os.getpid())
# p.nice(20)

path = '/home/pi/Desktop/web'

divx = 1
divy = 1

HEIGHT = 64 // divy
WIDTH = 128 // divx
h = HEIGHT
w = WIDTH
slide_y = h // 4
slide_x = w // 8

def load_model():

    json_file = open(os.path.join(path,'./model/model.json'), 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    opt = Adam(lr = 0.0001)
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(os.path.join(path,'./model/best_model.h5'))
    loaded_model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

    return loaded_model

md_pred = load_model()


def predict2(img):
    if type(img) == str:    img = cv2.imread(img,0)
    elif img.ndim > 2:  img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if img[0][0] > 1: img = img / 255.
    prediction = md_pred.predict(img.reshape(1, HEIGHT, WIDTH, -1))[0]
    return prediction
    

def segmentation(img, show=False, scaling=1, thres=0.5, expand=False):

    t1 = time()

    if type(img) == str:    img = cv2.imread(img,0)
    elif img.ndim > 2:  img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    backup = img.copy()
    img = cv2.resize(img,(img.shape[1]//(divx*scaling), img.shape[0]//(divy*scaling)))

    cv2.imshow('a', cv2.resize(img, (640//4, 480//4)))
    cv2.waitKey(100)

    x1 = 1e9
    x2 = 0
    y1 = 1e9
    y2 = 0
    cutout = -1
    n = 0

    for j in np.linspace(0,img.shape[0]-h,(img.shape[0]-h)//slide_y+1).astype('uint16'):
        for i in np.linspace(0,img.shape[1]-w,(img.shape[1]-w)//slide_x+1).astype('uint16'):

            pred = -1
            splice_ = img[j:j+h, i:i+w]

            pred = predict(splice_)
            pred = round(pred[0],2)

            if pred >= thres:

                cv2.imwrite('/home/pi/Desktop/web/split/'+str(random())+'.png', backup[j*divy:j*divy+h*divy, i*divx:i*divx+w*divx])
                 
                n+= 1

                if i < y1:  y1 = i
                if (i+w) > y2:  y2 = i+w
                if j < x1:  x1 = j
                if (j+h) > x2:  x2 = j+h

            if show:

                print(pred)
                cv2.imshow('x', cv2.resize(splice_, (640//4, 480//4)))
                cv2.waitKey(10)

    if n > 0:
    
        if expand:  e = 12
        else:   e = 0

        cutout = backup.copy()
        cutout = cutout[x1*(divy*scaling)-e:x2*(divy*scaling)+e,y1*(divx*scaling)-e*2:y2*(divx*scaling)+e*2]
        cv2.rectangle(backup, (y1*(divx*scaling)-e*2,x1*(divy*scaling)-e),(y2*(divx*scaling)+e*2,x2*(divy*scaling)+e),(0,255,0), 5)


    return backup, n, cutout, time() - t1

def get_plate(img,show=False):

    res = segmentation(img,scaling=2, show=show, thres=0.1)
    
    if res[1] > 0:
        
        res = segmentation(res[2], show=show)
   
    return res

if __name__ == "__main__":

    i = 'test.png'

    img = cv2.imread(i)

    res = segmentation(img, show=False, expand = True)
    plt.imshow(res[0])
    plt.show()
    plt.imshow(res[2])
    plt.show()
