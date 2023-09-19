#!/usr/bin/env python3

"""
MODEL INFERENCE SERVER
======================

Following program is used to perform inference on subject data using REST API
"""

# %%
# Importing Libraries
from inferenceGRPCClient import *
import os
import base64
from io import BytesIO
from PIL import Image
from flask import Flask, request, jsonify

# %%
# Execution
app = Flask(__name__)
resize = (int(os.environ.get('imageWidth', '640')), int(os.environ.get('imageHeight', '640')))
clientPool = dict()

@app.post(os.environ.get('apiRoute', '/person'))
def runner() -> tuple:
    imgStack = list()
    detecThresStack = list()
    nmsThresStack = list()
    iouThresStack = list()
    bxCusStack = list()
    trFramStack = list()
    reqDat = request.json
    if reqDat['clientID'] not in list(clientPool.keys()):
        clientPool[reqDat['clientID']] = personClient(serverIp = os.environ.get('grpcServerIP', '0.0.0.0:4393'), clientName=reqDat['clientID'])
    for timg, detecThres, nmsThres, iouThres, bxCus, trFram in zip(
        reqDat['images'],
        reqDat['detectionThreshold'],
        reqDat['nmsThreshold'],
        reqDat['iouThreshold'],
        reqDat['boxCushion'],
        reqDat['trackingFrames'],
    ):
        ti = base64.b64decode(timg.encode('utf-8'))
        byt1 = BytesIO(ti)
        img = np.asarray(Image.open(byt1))
        img = ocv.cvtColor(img, ocv.COLOR_RGB2BGR)
        img = ocv.resize(img, resize, interpolation = ocv.INTER_AREA)
        imgStack.append(img)
        detecThresStack.append(float(detecThres))
        nmsThresStack.append(float(nmsThres))
        iouThresStack.append(float(iouThres))
        bxCusStack.append(int(bxCus))
        trFramStack.append(int(trFram))
    ret = clientPool[reqDat['clientID']](imgStack, detecThresStack, nmsThresStack, iouThresStack, bxCusStack, trFramStack)
    return jsonify(ret), 200

if __name__ == '__main__':
    print("""
    =============================================================
    | Person Detection & Tracking Inference Server ( REST API ) |
    =============================================================
    Image Resize Target: {}
    GRPC Address: {}
    Inference Address: http://0.0.0.0:{}{}
    \n
    """.format(resize, os.environ.get('grpcServerIP', '0.0.0.0:4393'), int(os.environ.get('inferencePort', '8080')), os.environ.get('apiRoute', '/face')))
    app.run(host='0.0.0.0', port=int(os.environ.get('inferencePort', '8080')))
    