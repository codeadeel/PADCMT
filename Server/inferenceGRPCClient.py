#!/usr/bin/env python3

"""
MODEL INFERENCE CLIENT (GRPC)
=============================

Following program is used to perform inference on subject data using GRPC
"""

# %%
# Importing Libraries
import argparse
import string
import time
import json
import numpy as np
import cv2 as ocv
import grpc
import personCommunication_pb2
import personCommunication_pb2_grpc

# %%
# Main Inference Class
class personClient:
    def __init__(self, serverIp: string, idLen: int = 10, messageLen: int = 1000000000, clientName: string = None) -> None:
        """
        This method is used to initialize GRPC inference client

        Method Input
        =============
        serverIp : Server IP at which GRPC server is running
                            Format : "IP:Port"
                            Example : '0.0.0.0:1234'
        idLen : Length of client randomized id ( default : 10 )
        messageLen : Maximum size of send & receive messages
        clientName : Client Name / ID

        Method Output
        ==============
        None
        """
        self.serverIp = serverIp
        self.idLen = idLen
        self.serverOpts = [('grpc.max_send_message_length', messageLen), ('grpc.max_receive_message_length', messageLen)]
        self.channel = grpc.insecure_channel(self.serverIp, options = self.serverOpts)
        self.stub = personCommunication_pb2_grpc.personServiceStub(self.channel)
        self.client_name_chars = np.array(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'])
        if clientName==None:
            self.clientName = ''.join(np.random.choice(self.client_name_chars, size = self.idLen).tolist())
        else:
            self.clientName = clientName
        self.streamPool = list()
    
    def __inputProcessor__(self, inp1: np.array, detectionConfidence: list, nmsThreshold: list, iouThreshold: list, boxCushion: list, trackerFrames: list) -> personCommunication_pb2.serverInput:
        """
        This method is used to take parameters for processing & converts to GRPC input object

        Arguments
        =========
        inp1 : Image stack
        detectionConfidence : Bounding box detection confidence
        nmsThreshold : Non-Max-Suppression threshold
        iouThreshold : IOU threshold for ID tracker
        boxCushion : Bounding box cushion
        trackerFrames : Number of faces to track
        """
        if len(self.streamPool)!=len(inp1):
            self.streamPool = list()
            for i in range(len(inp1)):
                self.streamPool.append(''.join(np.random.choice(self.client_name_chars, size = self.idLen).tolist()))
        inp1Shape = inp1.shape
        return personCommunication_pb2.serverInput(
            imgs = inp1.tobytes(),
            batch = inp1Shape[0],
            width = inp1Shape[2],
            height = inp1Shape[1],
            channel = inp1Shape[3],
            dataType = inp1.dtype.name,
            detectionConfidence = json.dumps(detectionConfidence),
            nmsThreshold = json.dumps(nmsThreshold),
            iouThreshold = json.dumps(iouThreshold),
            boxCushion = json.dumps(boxCushion),
            trackerFrames = json.dumps(trackerFrames),
            trackerIds = json.dumps(self.streamPool),
            clientId = self.clientName
        )
    
    def __call__(self, imgList: list, detectionConfidence: list, nmsThreshold: list, iouThreshold: list, boxCushion: list, trackerFrames: list) -> string:
        """
        This method is used to handle inference requests & returns inference results

        Arguments
        =========
        x : List of OpenCV BGR images subject to required inference
        detectionConfidence : List of detection confidence
        nmsThreshold : NMS threshold for object detection
        iouThreshold : IOU threshold for object ID tracking
        boxCushion : Box-Cushion for object bounding boxes
        trackerFrames : Number of frames to track

        Outputs
        =======
        Person detection results
        """
        x = np.stack(imgList)
        resp = self.stub.inference(self.__inputProcessor__(x, detectionConfidence, nmsThreshold, iouThreshold, boxCushion, trackerFrames))
        return json.loads(resp.results)

    def __del__(self) -> None:
        """
        This method is used to close communication channel to GRPC server
        """
        self.channel.close()

# %%
# Client Execution
if __name__=='__main__':
    parser = argparse.ArgumentParser(description = 'Person Detection & Tracking Inference Client.')
    parser.add_argument('-l', '--link', type = str, help = 'Absolute Address of Subject Images', required = True)
    parser.add_argument('-ip', '--server_ip', type = str, help = 'IP Address to GRPC Server => IP:Port', required = True)
    parser.add_argument('-det', '--detectionThreshold', type = float, help = 'Person Detection Confidence Threshold', default = 0.6)
    parser.add_argument('-nms', '--nmsThreshold', type = float, help = 'Person Detection Non-Max Suppression', default = 0.5)
    parser.add_argument('-iou', '--iouThreshold', type = float, help = 'Person Detection IOU Tacker Threshold', default = 0.5)
    parser.add_argument('-bcus', '--boxCushion', type = int, help = 'Face Detection Box-Cushion', default = 0)
    parser.add_argument('-trF', '--trackerFrames', type = int, help = 'Face Detection Tracker Frames', default = 10)
    args = vars(parser.parse_args())
    personC = personClient(args['server_ip'])
    img = ocv.imread(args['link'])
    st = time.time()
    ret = personC([img], [args['detectionThreshold']], [args['nmsThreshold']], [args['iouThreshold']], [args['boxCushion']], [args['trackerFrames']])
    et = time.time() - st
    print(f'Inference Received!!! | Client ID: {personC.clientName} | Request Processing Time: {et}')
    print('-' * 30)
    print(ret)
