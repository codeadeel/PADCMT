# syntax=docker/dockerfile:1
#
# ===========================================================
# | PERSEON DETECTION & TRACKING INFERENCE SERVER (REST API)|
# ===========================================================
# 
# This Dockerfile is used to build model inference server for REST API based Person Detection & Tracking.
# 
# Quick Command to Build Inference Server
# =======================================
# docker build -t person:restserver -f BuildRestServer.Dockerfile .
#
# Quick Command to Run Inference Server
# =====================================
# docker run --rm -it \
#     -e grpcServerIP=0.0.0.0:4393  # [ Optional : GRPC Server IP ] \
#     -e imageWidth=512             # [ Optional : Target image width for transformstion ] \
#     -e imageHeight=512            # [ Optional : Target image height for transformstion ] \
#     -e inferencePort=8080         # [ Optional : Inference port for REST API ] \
#     -e apiRoute=/person             # [ Optional : Inference route for REST API ] \
#     person:restserver
#
# Main Build Script
# =================
#
# Pull ubuntu:latest Image from Docker-Hub
FROM ubuntu:latest
# Install Necessary Packages
RUN apt-get update ; apt-get install -y python3 python3-pip 
RUN pip3 install --no-cache-dir Pillow==9.4.0 opencv-python-headless==4.7.0.68 Flask==2.2.3 grpcio==1.56.2 protobuf==3.20.3
# Copy Server Files for Execution
WORKDIR /home
COPY ./Server/personCommunication_pb2.py ./personCommunication_pb2.py
COPY ./Server/personCommunication_pb2_grpc.py ./personCommunication_pb2_grpc.py
COPY ./Server/inferenceGRPCClient.py ./inferenceGRPCClient.py
COPY ./Server/inferenceRESTServer.py ./inferenceRESTServer.py
# Set Permissions & Create Execution Entrypoint
RUN chmod 777 ./inferenceRESTServer.py
ENTRYPOINT [ "./inferenceRESTServer.py" ]
