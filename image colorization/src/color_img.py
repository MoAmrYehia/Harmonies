import sys
import torch
import numpy as np
from PIL import Image
from skimage.transform import resize

from train import ConvNet
from utils import read_image, cvt2Lab, upsample, cvt2rgb
import matplotlib.pyplot as plt


MODEL_PATH              = "../model/image_colorization_model.pt"

def get_model():
    model = ConvNet()
    model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
    return model

def preprocess_image(inp_imgpath):
    img, real_size  = read_image(inp_imgpath)
    img_light       = cvt2Lab(img)[0]
    img_input       = np.expand_dims(img_light, axis=0)
    img_input       = np.expand_dims(img_input, axis=0)
    img_input       = torch.autograd.Variable(torch.from_numpy(img_input).float(), requires_grad=False)
    return img_light, img_input, real_size

def process_image(img, img_light):
    img = np.squeeze(img, axis=0)
    img = np.transpose(img.astype(np.float), (1, 2, 0))
    img = upsample(img)
    img = np.insert(img, 0, img_light, axis=2)
    img = (cvt2rgb(img) * 255.).astype(np.uint8)
    return img

def main(argv):
    if len(argv) != 2:
        print("Usage: python3 color_img.py input-image-path output-image-path")

    inp_imgpath, out_imgpath    = argv[0], argv[1]
    model                       = get_model()

    img_light, img_input, real_size = preprocess_image(inp_imgpath)

    img = model(img_input).cpu().data.numpy()
    img = process_image(img, img_light)

    img_result = Image.fromarray(img)
    img_result = img_result.resize((real_size[1], real_size[0]))
    img_result.save(out_imgpath, quality=90)
    print("Result image is saved to %s" % out_imgpath)
    return


if __name__ == '__main__':
    main(sys.argv[1:])

