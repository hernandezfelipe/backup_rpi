from imutils.video import WebcamVideoStream
import numpy as np
import cv2
from math import fabs
from datetime import datetime
from time import time, sleep
import cv2
import psutil
import os
from time import sleep

#p = psutil.Process(os.getpid())
#p.nice(0)

vs = WebcamVideoStream(src=0).start()

now = datetime.now()

pic_id = 0
for i in range(20): image = vs.read()
old_frame = None
diff = 0
changed = False

name = 'None.png' 

while True:

    if  datetime.now().hour >= 18 or datetime.now().hour <= 6:  sleep(600)

    else:
    
        t1 = time()

        image = vs.read()

        if old_frame is not None:
            
            frame = image[:,:,2]           

            diff = np.mean(cv2.subtract(cv2.resize(old_frame, (100,100)), cv2.resize(image, (100,100))))

            if diff > 4:

                if not changed:
                
                    changed = True                
                     
                    now = datetime.now()                    
                    name = '/home/pi/Desktop/web/raw/'+str(now.month)+'_'+str(now.day)+'_'+str(now.hour)+ '_'+str(now.minute)+'_'+str(now.second)+'_'                
                    cv2.imwrite(name+str(pic_id).zfill(10)+'.png', image)
                    print(name+str(pic_id).zfill(10)+'.png')

                else:
                                    
                    pic_id+=1
                    
                    cv2.imwrite(name+str(pic_id).zfill(10)+'.png', image)
                    print(name+str(pic_id).zfill(10)+'.png')
                
                cv2.imshow('web', cv2.resize(image, (640//4, 480//4)))
                cv2.waitKey(10)
                
                
            else:
            
                changed = False
                pic_id = 0

        old_frame = image

        #while time() - t1 < 1/240: pass

        #cv2.imshow('x', image)
        #cv2.waitKey(1)

        #print(diff)

"""

    cv2.imshow('x',temp)
    key = cv2.waitKey(1)

    if key == ord('q') and br > 0.01:
        br -= 0.01
        vs.set(10, br) #
        print('br', br)
    elif key == ord('e') and br < 0.8:
        br += 0.01
        vs.set(10, br) #
        print('br', br)
    elif key == ord('a') and cont > 0.01:
        cont -= 0.01
        vs.set(11, cont) #
        print('cont', cont)
    elif key == ord('d') and cont < 0.8:
        cont += 0.01
        vs.set(11, cont) #
        print('cont', cont)
    elif key == ord('z') and sat > 0.01:
        sat -= 0.01
        vs.set(12, sat) #
        print('sat', sat)
    elif key == ord('c') and sat < 0.8:
        sat += 0.01
        vs.set(12, sat) #
        print('sat', sat)

    print(diff, time()-t1, temp.shape)

"""

#cv2.destroyAllWindows()
vs.stop()




