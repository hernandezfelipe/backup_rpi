import os
from keras.models import model_from_json
from keras.optimizers import Adam
import cv2

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

path = os.path.dirname(__file__)

HEIGHT = 64
WIDTH = 128

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

def predict(img):
    if type(img) == str:    img = cv2.imread(img,0)
    elif img.ndim > 2:  img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if img[0][0] > 1: img = img / 255.
    img = cv2.resize(img, (WIDTH,HEIGHT))
    prediction = md_pred.predict(img.reshape(1, HEIGHT, WIDTH, -1))[0]
    return prediction
    

if __name__ == "__main__":

    for i in range(9):
    
        img = cv2.imread('./data/'+str(i)+'.png')
        cv2.imshow('img', img)
        cv2.waitKey(100)
        res = predict(img)
        print(res)
    

