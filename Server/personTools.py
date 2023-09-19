#!/usr/bin/env python3

"""
MODEL INFERENCE TOOLS
=====================

Following program is provides the tools for subject inference
"""

# %%
# Importing Libraries
import os
import string
import copy
from io import BytesIO
import numpy as np
import torch
import torchvision as tv
import dlib
import cv2 as ocv
from PIL import Image
import onnxruntime as ort

# %%
# Image Definition for Inference Model
class imgData:
    def __init__(
        self,
        dat: np.array,
        trackerId: string,
        detectionThreshold: float = 0.5,
        nmsThreshold: float = 0.5,
        boxCushion: int = 0,
        frames2Track: int = 30
    ) -> None:
        """
        This method initializes the image class, which processes the image according to model requirements

        Arguments
        =========
        dat : OpenCV image for inference
        trackerId : Stream Id, from which image is coming from, for tracking purposes ( Can be random, but should be same for data stream identification)
        detectionThreshold : Bounding box confidence threshold
        nmsThreshold : Non-Max-Suppression threshold in bounding box detection
        boxCushion : Bounding box padding pixels, which are added to bounding box dimensions
        frames2Track : Number of frames of the stream to track
        """
        self.rawData = dat
        self.data = ocv.cvtColor(self.rawData, ocv.COLOR_BGR2RGB)
        self.trackerId = trackerId
        self.detectionThreshold = detectionThreshold
        self.nmsThreshold = nmsThreshold
        self.boxCushion = boxCushion
        self.trackerFrames = frames2Track
        self.dataShape = self.data.shape
        self.targetWidth = int(os.environ.get('targetWidth', 640))
        self.targetHeight = int(os.environ.get('targetHeight', 640))
        if (self.dataShape == (self.targetWidth, self.targetHeight, 3)):
            self.inputTensor = self.data.transpose(2, 0, 1) / 255
        else:
            self.inputTensor = ocv.resize(self.data, (self.targetWidth, self.targetHeight)).transpose(2, 0, 1) / 255

# Model Inference Class
class Inference:
    def __init__(self) -> None:
        """
        This method initializes model inference class
        """
        self.targetWidth = int(os.environ.get('targetWidth', 640))
        self.targetHeight = int(os.environ.get('targetHeight', 640))
        self.detectionModelAddr = os.environ.get('detectionModel', '/workspace/personDetect_Nx3x640x640.onnx')
        try:
            self.detectionSession  = ort.InferenceSession(self.detectionModelAddr, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        except:
            print('DETECTION SESSION : Yikes! CUDA is not available. Falling back to CPU')
            self.detectionSession  = ort.InferenceSession(self.detectionModelAddr, providers=['CPUExecutionProvider'])
    
    def __call__(self, imgList):
        """
        This method performs inference on the provided image class object pool

        Arguments
        =========
        imgList : Image object pool, specifically designed for model inference

        Outputs
        =======
        finalBBOXsRaw : Bounding boxes as [ Box Probability, x1, y1, x2, y2 ], with respect to full dimension image
        """
        finalBBOXsRaw = list()
        
        inputTensor = np.array([i.inputTensor for i in imgList])
        personDetections = self.detectionSession.run(None, {"rgb640x640": inputTensor.astype(np.float32)})
        
        for bboxes, confidence, img in zip(personDetections[0], personDetections[1][:, :, 0], imgList):
            fconf = confidence[confidence>img.detectionThreshold][:, np.newaxis]
            pbbox = bboxes[confidence>img.detectionThreshold]
            pbbox[:, [0, 2]] /= self.targetWidth
            pbbox[:, [1, 3]] /= self.targetHeight
            pbbox[:, [0, 2]] *= img.dataShape[1]
            pbbox[:, [1, 3]] *= img.dataShape[0]
            idx = tv.ops.nms(torch.from_numpy(pbbox), torch.from_numpy(fconf[:, 0]), img.nmsThreshold).tolist()
            pbbox = pbbox[idx]
            fconf = fconf[idx]
            probWBox = np.concatenate((fconf, pbbox), axis=1)
            probWBox[:, 1:3] -= img.boxCushion
            probWBox[:, 3:5] += img.boxCushion
            probWBox[:, 1:3] = np.where(probWBox[:, 1:3]<0, 0, probWBox[:, 1:3])
            probWBox[:, 3] = np.where(probWBox[:, 3]>img.dataShape[1], img.dataShape[1], probWBox[:, 3])
            probWBox[:, 4] = np.where(probWBox[:, 4]>img.dataShape[0], img.dataShape[0], probWBox[:, 4])
            if len(probWBox)==0:
                finalBBOXsRaw.append([])
                continue
            else:
                finalBBOXsRaw.append(probWBox.tolist())
        return finalBBOXsRaw
    
# %%
# Stream Handler
class streamClient:
    def __init__(self, clientId: string) -> None:
        """
        This method is used to handle stream client for trackers

        Arguments:
        ==========
        cliendId : A consistent randomly generated client ID
        """
        self.clientId = clientId
        self.trackers = dict()
    
    def __idHandler__(self, trackerId):
        """
        This method implements the ID tracker for Person Detection

        Arguments:
        ==========
        trackerId : Tracker ID under consideration
        """
        currBox = copy.deepcopy(self.trackers[trackerId]['currentBBox'])
        prevBox = copy.deepcopy(self.trackers[trackerId]['prevBBox'])
        if ((len(prevBox)==0) and (len(currBox)==0) and (self.trackers[trackerId]['maxId']==0)):
            pass
        elif ((len(prevBox)==0) and (len(currBox)==0) and (self.trackers[trackerId]['maxId']>0)):
            self.trackers[trackerId]['idPool'] = list()
        elif ((len(prevBox)>0) and (len(currBox)==0) and (self.trackers[trackerId]['maxId']>0)):
            self.trackers[trackerId]['idPool'] = list()
        elif ((len(prevBox)==0) and (len(currBox)>0) and (self.trackers[trackerId]['maxId']==0)):
            self.trackers[trackerId]['idPool'] = (np.arange(len(currBox)) + 1).tolist() 
            self.trackers[trackerId]['maxId'] = max(self.trackers[trackerId]['idPool'])
        elif ((len(prevBox)==0) and (len(currBox)>0) and (self.trackers[trackerId]['maxId']>0)):
            self.trackers[trackerId]['idPool'] = (np.arange(len(currBox)) + 1 + self.trackers[trackerId]['maxId']).tolist()
            self.trackers[trackerId]['maxId'] = max(self.trackers[trackerId]['idPool'])
        elif ((len(prevBox)>0) and (len(currBox)>0) and (self.trackers[trackerId]['maxId']==0)):
            self.trackers[trackerId]['idPool'] = (np.arange(len(currBox)) + 1).tolist() 
            self.trackers[trackerId]['maxId'] = max(self.trackers[trackerId]['idPool'])
        else:
            currBox = torch.Tensor(currBox)
            prevBox = torch.Tensor(prevBox)
            iouer = tv.ops.box_iou(
                torch.where(currBox<0, 0, currBox),
                torch.where(prevBox<0, 0, prevBox)
            )
            iouer = torch.where(iouer >= self.trackers[trackerId]['iouT'], iouer, 0)
            newIds = torch.where(torch.sum(iouer, dim=1)>0, False, True)
            maxer = torch.argmax(iouer, dim=1)
            idInternal = list()
            for j, k in zip(maxer, newIds):
                if k==False:
                    idInternal.append(self.trackers[trackerId]['idPool'][j])
                else:
                    self.trackers[trackerId]['maxId'] += 1
                    idInternal.append(self.trackers[trackerId]['maxId'])
            self.trackers[trackerId]['idPool'] = idInternal
    
    def __call__(self, inferenceObj: Inference, imgList: list, detectionConfidence: list, nmsThreshold: list, iouThreshold: list, boxCushion: list, trackerFrames: list, trackerIds: list) -> dict:
        """
        This method is used to run the inference on the given stream

        Arguments:
        ==========
        inferenceObj : Inference model object to get raw faces inference
        imgList : Image object pool, specifically designed for model inference
        detectionConfidence : Bounding box confidence threshold
        nmsThreshold : Non-Max-Suppression threshold in bounding box detection
        iouThreshold : IOU threshold for object ID tracking
        boxCushion : Bounding box padding pixels, which are added to bounding box dimensions
        trackerFrames : Number of frames of the stream to track
        trackerIds : Unique consistent random tracker Ids to track frames in a stream
        """
        for img, dC, nmsT, bC, tF, tI, iT in zip(imgList, detectionConfidence, nmsThreshold, boxCushion, trackerFrames, trackerIds, iouThreshold):
            for i in list(self.trackers.keys()):
                if i not in trackerIds:
                    del self.trackers[i]
                    torch.cuda.empty_cache()
            if tI in list(self.trackers.keys()):
                self.trackers[tI]['counter'] += 1
                self.trackers[tI]['toInfer'] = False
                if (self.trackers[tI]['counter'] > tF) or (self.trackers[tI]['f2T']!=tF) or (self.trackers[tI]['iouT']!=iT):
                    self.trackers[tI] = {
                        'counter': 0,
                        'toInfer': True,
                        'f2T': tF,
                        'iouT': iT,
                        'image': imgData(img, tI, dC, nmsT, bC, tF),
                        'currentBBox': list(),
                        'prevBBox': copy.deepcopy(self.trackers[tI]['currentBBox']),
                        'maxId': self.trackers[tI]['maxId'],
                        'idPool': self.trackers[tI]['idPool'],
                        'currentBoxProbability': list(),
                        'trackerPool': list()
                    }
                else:
                    self.trackers[tI]['image'] = imgData(img, tI, dC, nmsT, bC, tF)
                    self.trackers[tI]['prevBBox'] = copy.deepcopy(self.trackers[tI]['currentBBox'])
            else:
                self.trackers[tI] = {
                    'counter': 0,
                    'toInfer': True,
                    'f2T': tF,
                    'iouT': iT,
                    'image': imgData(img, tI, dC, nmsT, bC, tF),
                    'currentBBox': list(),
                    'prevBBox': list(),
                    'maxId': 0,
                    'idPool': list(),
                    'currentBoxProbability': list(),
                    'trackerPool': list()
                }
        toInferImgs = dict()
        for i in list(self.trackers.keys()):
            if self.trackers[i]['toInfer'] == True:
                toInferImgs[i] = self.trackers[i]['image']
            else:
                for p in range(len(self.trackers[i]['trackerPool'])):
                    self.trackers[i]['trackerPool'][p].update(self.trackers[i]['image'].rawData)
                    newRect = self.trackers[i]['trackerPool'][p].get_position()
                    self.trackers[i]['currentBBox'][p][0] = newRect.left()
                    self.trackers[i]['currentBBox'][p][1] = newRect.top()
                    self.trackers[i]['currentBBox'][p][2] = newRect.right()
                    self.trackers[i]['currentBBox'][p][3] = newRect.bottom()
        if len(toInferImgs)>0:
            finalBBOXsRaw = inferenceObj(list(toInferImgs.values()))
            for i, keyId in zip(finalBBOXsRaw, list(toInferImgs.keys())):
                self.trackers[keyId]['currentBoxProbability'] = [fp[0] for fp in i]
                self.trackers[keyId]['currentBBox'] = [fb[1:] for fb in i]
                for t in self.trackers[keyId]['currentBBox']:
                    ttracker = dlib.correlation_tracker()
                    ttracker.start_track(self.trackers[keyId]['image'].rawData, dlib.rectangle(int(t[0]), int(t[1]), int(t[2]), int(t[3])))
                    self.trackers[keyId]['trackerPool'].append(ttracker)
        outJson = dict()
        for i in trackerIds:
            outBBox = list()
            outBBoxProbability = list()
            outIds = list()
            tbox = list()
            self.__idHandler__(i)
            for cbox in self.trackers[i]['currentBBox']:
                tbox.append([
                    cbox[0] / self.trackers[i]['image'].dataShape[1],
                    cbox[1] / self.trackers[i]['image'].dataShape[0],
                    cbox[2] / self.trackers[i]['image'].dataShape[1],
                    cbox[3] / self.trackers[i]['image'].dataShape[0]
                ])
            outBBox.append(tbox)
            outBBoxProbability.append(self.trackers[i]['currentBoxProbability'])
            outIds.append(self.trackers[i]['idPool'])
            outJson[i] = {
                'bbox': outBBox,
                'boxProbability': outBBoxProbability,
                'trackerIds': outIds
            }
        return outJson
    
if __name__=="__main__":
    print('PERSON DETECTION & TRACKING MODULE')
