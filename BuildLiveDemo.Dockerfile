# syntax=docker/dockerfile:1
#
# =========================================
# | PERSON DETECTION & TRACKING LIVE DEMO |
# =========================================
# 
# This Dockerfile is used to build live demo app for Person Detection & Tracking.
#
# Quick Command to Build Live Demo
# ================================
# docker build -t person:livedemo -f BuildLiveDemo.Dockerfile .
#
# Quick Command to Run Live Demo
# ==============================
# docker run --rm -it \
#     -e serverIP='http://172.17.0.5:8080'              # [ Required : Person Server Inference API ] \
#     -e pageUpdateInterval=1000                        # [ Optional : Page Update Interval ] \
#     person:livedemo
#
# Main Build Script
# =================
#
# Pull ubuntu:latest Image from Docker-Hub
FROM ubuntu:latest
# Install Necessary Packages
RUN apt-get update ; apt-get install -y --fix-missing ; apt-get install -y python3 python3-pip libgl1-mesa-dev libglib2.0-0 libgtk-3-dev
RUN pip3 install opencv-python-headless==4.7.0.68 streamlit==1.21.0 streamlit-webrtc==0.45.1 altair==4.2.2 grpcio==1.56.2 bounding-box==0.1.3 streamlit-autorefresh==1.0.1
# Copy Resources to Respective Directories
RUN mkdir /root/.streamlit
COPY ./demoApp/liveDemoConfig.toml /root/.streamlit/config.toml
WORKDIR /home
COPY ./Server/personCommunication_pb2.py ./personCommunication_pb2.py
COPY ./Server/personCommunication_pb2_grpc.py ./personCommunication_pb2_grpc.py
COPY ./Server/inferenceGRPCClient.py ./inferenceGRPCClient.py
COPY ./demoApp/logo.png ./logo.png
COPY ./demoApp/liveDemo.py ./demo.py
# Set Permissions & Create Execution Entrypoint
RUN chmod 777 ./demo.py
ENTRYPOINT ["streamlit", "run", "./demo.py"]
