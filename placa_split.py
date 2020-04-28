import os
from model_full import segmentation, get_plate
import cv2
from random import randint
from time import sleep
import psutil
from pathlib import Path

raw = '/home/pi/Desktop/web/raw/'
cropped = '/home/pi/Desktop/web/cropped/'

while True:
    
    paths = sorted(Path(raw).iterdir(), key=os.path.getmtime)
    lista = [str(f) for f in paths]

    if len(lista) == 0:

        print("Empty dir - placa_split")
        sleep(10)

    else:

        for i in lista:

            try:

                img = cv2.imread(i)
                res = get_plate(img)
                #res = segmentation(img)
                print(i, res[1], res[-1])

                if res[1] >= 1:
                
                    name = i.replace(raw,cropped)
                    ret = cv2.imwrite(name, res[2])
                    cv2.imshow('x', cv2.resize(res[0], (640//4, 480//4)))  
                    cv2.waitKey(100)                                         

            except Exception as e:

                print(e)

            try:

                os.remove(i)

            except Exception as e:

                print(e)

