# -*- coding: utf-8 -*-

from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np
import tensorflow as tf

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

# Keras
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from tensorflow.keras.applications.imagenet_utils import decode_predictions
from werkzeug.utils import secure_filename
#from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH ='model_mobilenetv2.h5'

# Load your trained model
model = MobileNetV2(weights = MODEL_PATH)



import numpy as np 

import re
import base64

import numpy as np

from PIL import Image
from io import BytesIO


def base64_to_pil(img_base64):
	"""
	Convert base64 image data to PIL image
	"""
	image_data = re.sub('^data:image/.+;base64,', '', img_base64)
	pil_image = Image.open(BytesIO(base64.b64decode(image_data)))
	return pil_image


def np_to_base64(img_np):
	"""
	Convert numpy image (RGB) to base64 string
	"""
	img = Image.fromarray(img_np.astype('uint8'), 'RGB')
	buffered = BytesIO()
	img.save(buffered, format="PNG")
	return u"data:image/png;base64," + base64.b64encode(buffered.getvalue()).decode("ascii")



#def model_predict(img_path, model):
def model_predict(img, model):
	#print(img_path)
	#img = image.load_img(img_path, target_size=(224, 224))
	
	#img = img.resize((224,224))
	img = np.resize(img, (224,224,3))
	print(img.shape)
	print('-'*20)
	print(img)
	# Preprocessing the image
	x = image.img_to_array(img)
	x = x.astype('float32')
	print(x.shape)
	# x = np.true_divide(x, 255)
	## Scaling
	x=x/255.
	x = np.expand_dims(x, axis=0)
	#print(x.shape)
   

	# Be careful how your trained model deals with the input
	# otherwise, it won't make correct prediction!
   # x = preprocess_input(x)

	preds = model.predict(x)
	#print(preds)
	#preds=np.argmax(preds, axis=1)
	"""if preds==0:
		preds="The leaf is a diseased cotton leaf"
	elif preds==1:
		preds="The leaf is a diseased cotton plant"
	elif preds==2:
		preds="The leaf is a fresh cotton leaf"
	else:
		preds="The leaf is a fresh cotton plant"
	"""
	pred_proba = "{:.3f}".format(np.amax(preds))
	pred_class = decode_predictions(preds, top = 1)
	
	result = str(pred_class[0][0][1])
	result = result.replace('_', ' ').capitalize()
	
	return f"The result is {result}"


@app.route('/', methods=['GET'])
def index():
	# Main page
	return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
	if request.method == 'POST':
		# Get the file from post request
		#f = request.files['file']
		#print(request.files)
		#print('-' * 20)
		filestr = request.files['file'].read()
		print(filestr)
		import cv2
		npimg = np.fromstring(filestr, np.uint8)
		print(npimg)
		print(npimg.shape)
		#img = cv2.imdecode(npimg, cv2.CV_LOAD_IMAGE_UNCHANGED)
		img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
		#img = base64_to_pil(request.files['file'].read())
		print(img)
		print(img.shape)
		# Save the file to ./uploads
		#basepath = os.path.dirname(__file__)
		#file_path = os.path.join(
		#    basepath, 'uploads', secure_filename(f.filename))
		#f.save(file_path)
		
		# Make prediction
		#preds = model_predict(file_path, model)
		preds= model_predict(img, model)
		result=preds
		return result
	return None


if __name__ == '__main__':
	app.run(port=5001,debug=True)
