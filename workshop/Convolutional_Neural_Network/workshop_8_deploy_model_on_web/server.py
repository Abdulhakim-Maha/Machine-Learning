import numpy as np
import json
from flask import Flask, request
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps

model = load_model('./digit_model.h5')
print('model: ', model.get_config()['layers'][0]['config']['batch_input_shape'])

app = Flask(__name__)
@app.route('/model', methods=['POST'])
def run_model():
	req_data = request.get_json(force=True)
	image_data = req_data['img']
	img = np.array(image_data).reshape(28, 28, 1)

	image = np.array(image_data).reshape(1, 28, 28, 1)
	pred = model.predict(image)
	# print(pred)
	digit = np.argmax(pred)
	# print(digit)
	return str(digit)

if __name__ == '__main__':
	app.run()
