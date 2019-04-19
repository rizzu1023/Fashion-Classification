
import pickle
import cv2
import os
# from flaskblog.core import finalized_model.sav

def classifier(image_path):
    filename = 'finalized_model.sav'

    # cwd = os.getcwd()


    image = cv2.imread(r"C:\Users\Rizzu1023\Desktop\ML-MP\sample\unnamed23.jpg", 0)
    res = cv2.resize(image, dsize=(28, 28), interpolation=cv2.INTER_CUBIC)
    res=res.reshape(1,28,28,1)
    # file_path = "ML-MP\flaskblog\core\finalized_model.sav"

    loaded_model = pickle.load(open(filename, 'rb'))
    # result = loaded_model.predict_classes(res)
    result = loaded_model.cnn_model.predict_classes(res)
    return result[0]
#cnn_model.predict_classes(res)
