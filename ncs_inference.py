from openvino.inference_engine import IENetwork, IEPlugin, IECore
import cv2
import numpy as np 

device = "MYRIAD"
model = './model/plate_model.xml'
model_bin = './model/plate_model.bin'
ie = IECore()
net = IENetwork(model=model, weights=model_bin) 
myriad_config = {'VPU_HW_STAGES_OPTIMIZATION': 'NO'}
ie.set_config(myriad_config, "MYRIAD")  
plugin = IEPlugin(device)
input_blob = next(iter(net.inputs))
exec_net = ie.load_network(network=net, num_requests=2, device_name=device)

HEIGHT = 64
WIDTH = 128

def predict(img):
    if type(img) == str:    img = cv2.imread(img,0)
    elif img.ndim > 2:  img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
    img = cv2.resize(img, (WIDTH,HEIGHT))
    prediction = exec_net.infer({'conv2d_1_input': img})
    prediction = prediction['dense_4/Sigmoid'][0]
    
    return prediction

 
if __name__ == "__main__":

    for i in range(9):
    
        img = cv2.imread('./data/'+str(i)+'.png')
        cv2.imshow('img', img)
        cv2.waitKey(100)
        res = predict(img)
        print(res)
    
