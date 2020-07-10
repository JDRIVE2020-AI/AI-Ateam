#!/usr/bin/env python

import argparse
import base64
import json
import time
import numpy as np
import socketio
import eventlet
import eventlet.wsgi
import scipy.misc
from io import BytesIO
from flask import Flask, render_template
from PIL import Image
from PIL import ImageOps

from keras.models import model_from_json
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import array_to_img, img_to_array

# Fix error with Keras and TensorFlow
import tensorflow as tf
#tf.python.control_flow_ops = tf

#csv 2018/10/03
import csv
#csv 2018/10/03 finish

sio = socketio.Server()
app = Flask(__name__)
model = None
prev_image_array = None

e_fst = 0.0
e_sec = 0.0

def gen_f():
    if not hasattr(gen_f, "flag"):
        gen_f.flag = True
        return True
    return False
    
def gen2_f():
    if not hasattr(gen2_f, "flag2"):
        gen2_f.flag = True
        return True
    return False

def gen_f2():
    global cur_time, ref_time
    cur_time = time.time()
    if cur_time - ref_time > 1:
        flag3 = True
        ref_time = cur_time
    else:
        flag3 = False
    return flag3

def show_image():
    global ref_time#, process_image
    image = Image.open('/home/deepstation/AI_training/reverse/images/center_g.jpg')
    #process_image.LOAD_TRUNCATED_IMAGES = True
    if gen_f():
        ref_time = time.time()
        _ = plt.imshow(image)
        plt.title('Center')
        plt.show()
    if gen_f2():
        _ = plt.imshow(image)
        plt.title('Center')
        plt.show()

def process_image(image):
    image = np.array(image, np.float32)
    image = image[60:140, :]
    image = scipy.misc.imresize(image, (66, 200))
    return image

@sio.on('telemetry')
def telemetry(sid, data):
    #define deflection 
    global e_fst
    global e_sec
    #csvlist Initialization
    csvlist = []
    
    # The current steering angle of the car
    steering_angle = data["steering_angle"]
    # The current throttle of the car
    throttle = data["throttle"]
    # The current speed of the car
    speed = data["speed"]

    # The current image from the center camera of the car
    image_path = data["image"]
    image = Image.open(BytesIO(base64.b64decode(image_path)))
    image_array = np.asarray(image)

    # Preprocess image
    image_array = process_image(image_array)
    transformed_image_array = image_array[None, :, :, :]

    steering_angle = float(model.predict(
        transformed_image_array, batch_size=1))
    # Counteract for model's bias towards 0 values
    steering_angle = steering_angle * 1
    # The driving model currently just outputs a constant throttle. Feel free
    # to edit this.
    throttle = 0.1
    if float(speed) < 10:
        throttle = 0.5
        
    print('Steering Angle: %f, \t Throttle: %f' % (steering_angle, throttle))
    send_control(steering_angle, throttle)
    
@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    send_control(0, 0)


def send_control(steering_angle, throttle):
    sio.emit("steer", data={
        'steering_angle': steering_angle.__str__(),
        'throttle': throttle.__str__()
    }, skip_sid=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument('model', type=str,
                        help='Path to model definition json. Model weights should be on the same path.')
    args = parser.parse_args()
    with open(args.model, 'r') as jfile:
        model = model_from_json(json.load(jfile))

    model.compile("adam", "mse")
    weights_file = args.model.replace('json', 'h5')
    model.load_weights(weights_file)

    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
