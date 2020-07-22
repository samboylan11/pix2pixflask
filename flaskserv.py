import base64
import numpy as np
import io
from PIL import Image
import keras
from keras import backend as K
from keras.models import Sequential
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array

import time

from flask import request
from flask import jsonify
from flask import Flask

from matplotlib.image import imsave


app = Flask(__name__)

def get_model():
    global model
    model = load_model('shoes_model.h5')
    print(' * Model Loaded!')
    
def preprocess_image(image, target_size):
#     if image.mode != 'RGBZZ"
    image = image.resize(target_size)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    
    return image

print(' * Loading Keras model...')
get_model()

@app.route('/generate', methods=['GET','POST'])
def generate():
    message = request.get_json(force=True)
    encoded = message['image']
    decoded = base64.b64decode(encoded)
    image = Image.open(io.BytesIO(decoded))
    processed_image = preprocess_image(image, target_size=(256, 256))
    
    prediction = model.predict(processed_image)
    
    epoch_time = int(time.time())
    outputfile = 'output_%s.png' %(epoch_time)
    imsave(outputfile, prediction.numpy())
    response = {'result':outputfile}
    
    return jsonify(response)

@app.route('/download/<fname>', methods=['GET'])
def download(fname):
    return send_file(fname)
    
#     messsage = request.get_json(force=True)
#     name = message['name']
#     response = {'greeting':'hello, ' + name + '!'}
#     return jsonify(response)

#     return 'Flask is running'