# syntax=docker/dockerfile:1
#
# ======================================================
# | PERSON DETECTION & TRACKING INFERENCE SERVER (GRPC)|
# ======================================================
# 
# This Dockerfile is used to build model inference server for GRPC based Person Detection & Tracking.
# 
# Quick Command to Build Inference Server
# =======================================
# docker build -t person:grpcserver -f BuildGRPCServer.Dockerfile .
#
# Quick Command to Run Inference Server
# =====================================
# docker run --rm -it --gpus all \
#     -e targetWidth=640                                        # [ Optional : Target image width for detection inference ] \
#     -e targetHeight=640                                       # [ Optional : Target image height for detection inference ] \
#     -e serverIp=[::]:8080                                     # [ Optional : Server IP for GRPC Server ] \
#     -e messageLength=1000000000                               # [ Optional : Message length for GRPC Server ] \
#     -e numWorkers=1                                           # [ Optional : Number of workers for GRPC Server ] \
#     -v [ Required: Your Trained ONNX Model for Person Detection ]:/personDetect_Nx3x640x640.onnx:ro \
#     person:grpcserver
#
# Main Build Script
# =================
#
# Pull pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime Image from DOCKER-HUB
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime
# Install Necessary Packages
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update --fix-missing ; apt-get install -y build-essential
RUN pip3 install --no-cache-dir opencv-python-headless==4.8.0.74 opencv-contrib-python-headless==4.8.0.74 onnxruntime-gpu==1.15.1 grpcio==1.56.2 dlib==19.24.2
# Copy Server Files for Execution
COPY ./Server/personCommunication_pb2.py ./personCommunication_pb2.py
COPY ./Server/personCommunication_pb2_grpc.py ./personCommunication_pb2_grpc.py
COPY ./Server/personTools.py ./personTools.py
COPY ./Server/personServer.py ./personServer.py
# Set Permissions & Create Execution Entrypoint
RUN chmod 777 ./personServer.py
ENTRYPOINT [ "./personServer.py" ]
