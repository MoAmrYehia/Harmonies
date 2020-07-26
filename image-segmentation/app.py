import io
import numpy as np
from PIL import Image
import os
import json
import cv2
import random
import base64
from flask import Flask, jsonify
from flask import request
from flask_cors import CORS
import torch
import torchvision

import detectron2

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog

cfg = get_cfg()
# add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
cfg.merge_from_file(model_zoo.get_config_file(
    "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
# Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
    "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")

cfg.MODEL.DEVICE = 'cpu'

predictor = DefaultPredictor(cfg)


app = Flask(__name__)
CORS(app)

@app.route("/", methods=['GET', 'POST', 'OPTIONS'])
def index():
    req = request.get_json()
    if type(req) is dict:
        request_string = req['img']
    else:
        return "Invalid Request"
    # return str(request.get_json())
    key = "base64,"
    index = request_string.find(key)
    if(index != -1):
        original_mime = request_string[:index+len(key)]
        req = request_string[index+len(key):]
    else:
        original_mime = ""
        req = request_string

    input_image = base64.b64decode(req)
    im = Image.open(io.BytesIO(input_image))
    im = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)
    # return request.json.get("img")
    outputs = predictor(im)

    classes = outputs["instances"].pred_classes.tolist()
    persons = np.array([np.array(outputs["instances"].pred_masks[x].cpu())
                        for x in range(len(classes)) if classes[x] == 0])

    if(len(persons) == 0):
        return jsonify({"res": original_mime + req})

    persons = persons.sum(axis=0)
    persons = persons.clip(0, 1).reshape(
        persons.shape[0], persons.shape[1], 1).astype("uint8")

    img = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    img = np.dstack((img, persons*255))
    img = Image.fromarray(img)

    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    img.save("file.png")
    response = base64.b64encode(buffer.getvalue())
    return {"res": "data:image/png;base64," + str(response)[2:-1]}


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
