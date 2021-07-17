import os
import sys
import numpy as np
from PIL import Image
from keras.models import load_model


if os.path.isfile('./model/facenet_keras.h5'):
    print("Loading model...")
    facenet_model = load_model('./model/facenet_keras.h5')
else:
    print("Model not found. Please download")
    sys.exit(-1)

if not os.path.exists('dataset'):
    print("Please run createDataset.py first")
    sys.exit(-1)

if not os.path.exists('vectors'):
    os.makedirs('vectors')

for personName in os.listdir('./dataset/'):
    if not os.path.exists('vectors/' + personName):
        os.makedirs('vectors/' + personName)

    for filename in os.listdir('./dataset/' + personName + '/'):
        face_array = np.asarray(Image.open('./dataset/' + personName + '/' + filename).resize((160,160)).convert('RGB'), dtype='float32')
        mean, std = face_array.mean(), face_array.std()
        face_array = (face_array - mean) / std 
        yhat = facenet_model.predict(np.expand_dims(face_array, axis=0))

        np.save('vectors/' + personName + '/' + filename, yhat)
