import io
import numpy as np
from PIL import Image
import json
import cv2
import random
import base64
from flask import Flask, request, Response

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

@app.route("/", methods=['POST', 'OPTIONS'])
def segment():
    res = Response(headers={
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Headers": "Content-Type, X-Requested-With",
        "Access-Control-Allow-Methods": "POST, OPTIONS"
        })
    try:
        
        req = request.get_json()
        if type(req) is dict:
            request_string = req['img']
        else:
            res.data = "Invalid Request"
            return res
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
        im = np.asarray(Image.open(io.BytesIO(input_image)))
        # return request.json.get("img")
        if len(im.shape) > 2:
            if im.shape[2] > 3:
                im = im[:, :, :3]
        else:
            im = np.dstack((im, im, im))
        print(im.shape)
        # im, _ = read_image(req)

        outputs = predictor(im)

        classes = outputs["instances"].pred_classes.tolist()
        persons = np.array([np.array(outputs["instances"].pred_masks[x].cpu())
                            for x in range(len(classes)) if classes[x] == 0])

        if(len(persons) == 0):
            res.mimetype = "application/json"
            res.data = json.dumps({"res": original_mime + req})
            return res

        persons = persons.sum(axis=0)
        persons = persons.clip(0, 1).reshape(
            persons.shape[0], persons.shape[1], 1).astype("uint8")

        im = np.dstack((im, persons*255))
        im = Image.fromarray(im)

        buffer = io.BytesIO()
        im.save(buffer, format="PNG")
        response = base64.b64encode(buffer.getvalue())
        res.mimetype = "application/json"
        res.data = json.dumps({"res": "data:image/png;base64," + str(response)[2:-1]})
        return res
    except Exception as e:
        res.data = traceback.format_exc()
        return res


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
