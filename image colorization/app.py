from flask import Flask, request, Response
import json
import base64
import sys
import torch
import skimage
from skimage.transform import rescale, resize
import numpy as np
import io
from PIL import Image

# from train import ConvNet
# from utils import read_image, cvt2Lab, upsample, cvt2rgb


class ConvNet(torch.nn.Module):
    """convolutional network that performs image colorization"""
    def __init__(self, num_classes=10):
        super(ConvNet, self).__init__()
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, kernel_size=5, padding=2),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(32),
            torch.nn.MaxPool2d(kernel_size=2))

        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 8, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(8),
            torch.nn.MaxPool2d(kernel_size=2))

        self.layer3 = torch.nn.Conv2d(8, 2, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x



MODEL_PATH = "./model/image_colorization_model.pt"

def read_image(b64, size=(256, 256), training=False):
    if isinstance(b64, bytes):
        b64 = b64.decode("utf-8")
    imgdata = base64.b64decode(b64)
    img = skimage.io.imread(imgdata, plugin='imageio')
    if len(img.shape) > 2 and img.shape[2] > 3:
        img = img[:, :, :3]
    real_size = img.shape
    if img.shape!=size and not training:
        img     = resize(img, size, anti_aliasing=False)
    if len(img.shape) == 2:
        img = np.stack([img, img, img], 2)
    return img, real_size[:2]

def process_image(img, img_light):
    img = np.squeeze(img, axis=0)
    img = np.transpose(img.astype(np.float), (1, 2, 0))
    img = upsample(img)
    img = np.insert(img, 0, img_light, axis=2)
    img = (cvt2rgb(img) * 255.).astype(np.uint8)
    return img

def get_model():
    model = ConvNet()
    model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
    return model

def preprocess_image(b64):
    img, real_size  = read_image(b64)
    img_light       = cvt2Lab(img)[0]
    img_input       = np.expand_dims(img_light, axis=0)
    img_input       = np.expand_dims(img_input, axis=0)
    img_input       = torch.autograd.Variable(torch.from_numpy(img_input).float(), requires_grad=False)
    return img_light, img_input, real_size

def cvt2Lab(image):
    Lab = skimage.color.rgb2lab(image)
    return Lab[:, :, 0], Lab[:, :, 1:]  # L, ab


def cvt2rgb(image):
    return skimage.color.lab2rgb(image)


def upsample(image):
    return rescale(image, 4, mode='constant', order=3)

model = get_model()

app = Flask(__name__)

@app.route("/", methods=["POST", "OPTIONS"])
def color():
    res = Response(headers={
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Headers": "Content-Type, X-Requested-With",
        "Access-Control-Allow-Methods": "POST, OPTIONS"
        })

    try:
        req = request.get_json()
        if type(req) is dict:
            inputb64 = req['img']
        else:
            res.data = "Invalid Request"
            return res
        
        key = "base64,"
        index = inputb64.find(key)
        if(index != -1):
            original_mime = inputb64[:index+len(key)]
            inputb64 = inputb64[index+len(key):]
        else:
            original_mime = ""
        
        img_light, img_input, real_size = preprocess_image(inputb64)
        img = model(img_input).cpu().data.numpy()
        img = process_image(img, img_light)

        img_result = Image.fromarray(img)
        img_result = img_result.resize((real_size[1], real_size[0]))

        buffer = io.BytesIO()
        img_result.save(buffer, format="PNG")
        response = base64.b64encode(buffer.getvalue())
        res.mimetype = "application/json"
        res.data = json.dumps({"res": "data:image/png;base64," + str(response)[2:-1]})
        return res
    except Exception as e:
        res.data = traceback.format_exc()
        return res
       

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)