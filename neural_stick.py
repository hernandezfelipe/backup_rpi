import os
import cv2

net = cv2.dnn_ClassificationModel('./model/plate_model.xml','./model/plate_model.bin')

net.setPreferableTarget(cv2.dnn.DNN_TARGET_MYRIAD)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

path = os.path.dirname(__file__)

HEIGHT = 32
WIDTH = 32

def predict(img):
    if type(img) == str:    img = cv2.imread(img,0)
    elif img.ndim > 2:  img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)   
    #if img[0][0] > 1:   img = img / 255.
    img = cv2.resize(img, (WIDTH,HEIGHT))
    prediction = net.predict(img)[0][0]
    
    return prediction

if __name__ == "__main__":

    for i in range(9):
    
        img = cv2.imread('./data/'+str(i)+'.png')
        cv2.imshow('img', img)
        cv2.waitKey(100)
        res = predict(img)
        print(res)
    


