
import pickle
import cv2
import os
from keras import backend as K

def classifier(image_name):
    filename = 'finalized_model.sav'
    model = pickle.load(open(filename, 'rb'))
    # cwd = os.getcwd()

    path = r"C:\Users\Rizzu1023\Desktop\ML-MP\flaskblog\static\profile_pics"
    name = "88fa542feb993d3f.jpg"
    loc = "{}\{}".format(path,image_name)

    dict = {0:"T-shirt/top",1:"Trouser",2:"Pullover",3:"Dress",4:"Coat",5:"Sandal",6:"Shirt",7:"Sneaker",8:"Bag",9:"Ankle boot"}
    #loc =loc.replace("1",str(i))
    img = cv2.imread(loc,0)
    img = cv2.resize(img, dsize=(28, 28))
    img = cv2.bitwise_not(img)

    predict = model.predict_classes(img.reshape(1,28,28,1),batch_size=150) 
    K.clear_session()
    return dict[predict[0]]


    
 


