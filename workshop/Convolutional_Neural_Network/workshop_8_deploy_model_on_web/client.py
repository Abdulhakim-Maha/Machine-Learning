import numpy as np
import json
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.image import rgb_to_grayscale
from PIL import ImageOps, Image
import requests

img = load_img('digit2.png', target_size=(28, 28))
img =ImageOps.invert(img)
img = img_to_array(img)
img = rgb_to_grayscale(img)
img = img / 255.0
img_lst = np.squeeze(img).tolist()
data = json.dumps({"img":img_lst})
url = 'http://127.0.0.1:5000/model'

response = requests.post(url, data)
print('Digit =', response.text)