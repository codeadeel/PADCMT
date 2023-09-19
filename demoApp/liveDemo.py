#!/usr/bin/env python3

'''
Person Detection & Tracking Live Demo App
==========================================

This App is used to run live demo using browser webcam
'''

# %%
# Importing Libraries
from inferenceGRPCClient import *
import os
from PIL import Image
from bounding_box import bounding_box as bb
import av
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode
from streamlit_autorefresh import st_autorefresh

# %%
# Execution
@st.cache_resource(show_spinner=True)
def oneTimeRunner():
    pC = personClient(os.environ.get('serverIP', '0.0.0.0:4343'))
    class heamap:
        def __init__(self):
            self.heatMapArr = None
            self.active = False
            self.heatChecker = False
            
    class peopleCounter:
        def __init__(self):
            self.trackerPool = dict()
            self.inCounter = 0
            self.outCounter = 0
            self.countDirection = ''
            self.x1 = 0
            self.y1 = 0
            self.x2 = 0
            self.y2 = 0
            self.personChecker = False
        
        def __checkIntersection__(self, s0, s1):
            dx0 = s0[1][0]-s0[0][0]
            dx1 = s1[1][0]-s1[0][0]
            dy0 = s0[1][1]-s0[0][1]
            dy1 = s1[1][1]-s1[0][1]
            p0 = dy1*(s1[1][0]-s0[0][0]) - dx1*(s1[1][1]-s0[0][1])
            p1 = dy1*(s1[1][0]-s0[1][0]) - dx1*(s1[1][1]-s0[1][1])
            p2 = dy0*(s0[1][0]-s1[0][0]) - dx0*(s0[1][1]-s1[0][1])
            p3 = dy0*(s0[1][0]-s1[1][0]) - dx0*(s0[1][1]-s1[1][1])
            return (p0*p1<=0) & (p2*p3<=0)
        
        def track(self, id, coords):
            if id not in list(self.trackerPool.keys()):
                self.trackerPool[id] = {
                    'currCoord': coords,
                    'prevCoord': None,
                    'cutFreq': 0
                }
            else:
                self.trackerPool[id]['prevCoord'] = self.trackerPool[id]['currCoord']
                self.trackerPool[id]['currCoord'] = coords
                currentCut = self.__checkIntersection__([self.trackerPool[id]['prevCoord'], self.trackerPool[id]['currCoord']], [(self.x1, self.y1), (self.x2, self.y2)])
                if ((currentCut == True) and (self.trackerPool[id]['cutFreq'] == 0)):
                    self.trackerPool[id]['cutFreq'] += 1
                    if (self.trackerPool[id]['prevCoord'][1] > self.trackerPool[id]['currCoord'][1]):
                        if (peC.countDirection == 'Going Up'):
                            self.inCounter += 1
                        if (peC.countDirection == 'Going Down'):
                            self.outCounter += 1
                    if (self.trackerPool[id]['prevCoord'][1] < self.trackerPool[id]['currCoord'][1]):
                        if (peC.countDirection == 'Going Up'):
                            self.outCounter += 1
                        if (peC.countDirection == 'Going Down'):
                            self.inCounter += 1
                    if (self.trackerPool[id]['prevCoord'][0] < self.trackerPool[id]['currCoord'][0]):
                        if (peC.countDirection == 'Going Right'):
                            self.inCounter += 1
                        if (peC.countDirection == 'Going Left'):
                            self.outCounter += 1
                    if (self.trackerPool[id]['prevCoord'][0] > self.trackerPool[id]['currCoord'][0]):
                        if (peC.countDirection == 'Going Right'):
                            self.outCounter += 1
                        if (peC.countDirection == 'Going Left'):
                            self.inCounter += 1
                if (self.trackerPool[id]['cutFreq'] >= 1):
                    if (self.trackerPool[id]['cutFreq']==cutFreq):
                        self.trackerPool[id]['cutFreq'] = 0
                    else:
                        self.trackerPool[id]['cutFreq'] += 1

        def prune(self, currIds):
            for id in list(self.trackerPool.keys()):
                if id not in currIds:
                    del self.trackerPool[id]

    hT = heamap()
    peC = peopleCounter()
    return pC, hT, peC

def counterReset():
    peC.inCounter = 0
    peC.outCounter = 0

def updateX1():
    peC.x1 = x1

def updateY1():
    peC.y1 = y1

def updateX2():
    peC.x2 = x2

def updateY2():
    peC.y2 = y2

def updateHeatChecker():
    hT.heatChecker = not heatChecker

def updatePersonChecker():
    peC.personChecker = not countCheck

logoData = '/home/logo.png'
st.set_page_config(page_title="Person Detection & Tracking", page_icon=logoData, layout="centered",initial_sidebar_state="collapsed")
grpcInferenceClient, hT, peC = oneTimeRunner()
activer = False

st_autorefresh(interval=int(os.environ.get('pageUpdateInterval', '1000')), key="personDet")

col1, col2 = st.columns(2)
with col1:
    st.image(Image.open(logoData))
with col2:
    st.header('Person Detection & Tracking')
    col11, col12, col13 = st.columns(3)
    with col11:
        st.metric('People In', str(peC.inCounter), str(0))
    with col12:
        st.metric('People Out', str(peC.outCounter), str(0), delta_color='inverse')
    with col13:
        st.metric('People Available', str(peC.inCounter - peC.outCounter), str(0))
st.markdown("---")

rtspAddr = st.sidebar.text_input('RTSP Stream Address', placeholder='rtsp://')
if ((len(rtspAddr)>0) and (activer == False)):
    activer = True
    vid = ocv.VideoCapture(rtspAddr)

st.sidebar.header('Person Tweaker')
detecThres = st.sidebar.slider('Detection Threshold', min_value=0.0, max_value=1.0, value=0.75, step=0.01)
nmsThres = st.sidebar.slider('Non-Max Suppression', min_value=0.0, max_value=1.0, value=0.5, step=0.01)
iouThres = st.sidebar.slider('IOU Tracker Threshold', min_value=0.0, max_value=1.0, value=0.5, step=0.01)
col3, col4 = st.sidebar.columns(2)
with col3:
    bxCus = st.number_input('Bounding Box Cushion', value=0)
    htPointer = st.number_input('Heatmap Pointer Size', min_value=0, value=10)
with col4:
    trFram = st.number_input('Tracking Frames', min_value=0, value=0)
    heatChecker = st.checkbox('Crowd Heatmaps', value=False, help='Display Presence Intensity', on_change=updateHeatChecker)
col5, col6 = st.sidebar.columns(2)
with col5:
    x1 = st.number_input('x1', min_value=0, value=0, on_change=updateX1)
    y1 = st.number_input('y1', min_value=0, value=0, on_change=updateY1) 
with col6:
    x2 = st.number_input('x2', min_value=0, value=0, on_change=updateX2)
    y2 = st.number_input('y2', min_value=0, value=0, on_change=updateY2)
col7, col8 = st.sidebar.columns(2)
with col7:
    peC.countDirection = st.radio('Person Counter Direction', ['Going Up', 'Going Down', 'Going Left', 'Going Right'], help='Direction of the Persons to Count For', on_change=counterReset)
with col8:
    cutFreq = st.number_input('Person Cut Frequency', min_value=0, value=10)
    countCheck = st.checkbox('Person Counter', value=False, help='Activate Person Counter', on_change=updatePersonChecker)

def frameProcessor(frame) -> av.VideoFrame:
    """
    This function serves as callback to process the subject frame from camera

    Arguments
    =========
    frame : Frame buffer received from the browser webcam

    Outputs
    =======
    Processed frame to display on browser
    """
    if activer:
        ret, img = vid.read()
        if ((len(img)==0) or (img==None)).any():
            img = frame.to_ndarray(format="bgr24")
    else:
        img = frame.to_ndarray(format="bgr24")
    if hT.heatChecker==True:
        if hT.active==False:
            hT.active = True
            hT.heatMapArr = np.zeros((img.shape[0], img.shape[1]))
    else:
        hT.heatMapArr = np.zeros((img.shape[0], img.shape[1]))
    if len(img)>0:
        img = ocv.flip(img, 1)
        imgShape = img.shape
        ret = grpcInferenceClient([img], [detecThres], [nmsThres], [iouThres], [bxCus], [trFram])
        currentInfer = list(ret.values())[0]
        boxesList = np.array(currentInfer['bbox'][0])
        trackerList = currentInfer['trackerIds'][0]
        if peC.personChecker==True:
            peC.prune(trackerList)
        if len(boxesList)>0:
            xCenter = (boxesList[:, 0] + ((boxesList[:, 2] - boxesList[:, 0]) / 2)) * imgShape[1]
            boxesList[:, [0, 2]] *= imgShape[1]
            boxesList[:, [1, 3]] *= imgShape[0]
            boxesList = boxesList.astype(np.int16)
            xCenter = xCenter.astype(np.int16)
            for b, x, t in zip(boxesList, xCenter, trackerList):
                if (peC.personChecker==True):
                    peC.track(t, (x, b[3]))
                try:
                    bb.add(img, b[0], b[1], b[2], b[3], 'Person ' + str(t), 'blue')
                    img = ocv.circle(img, (x, b[3]), 2, (0, 0, 255), 2)
                    if ((peC.personChecker == True) and (peC.trackerPool[t]['prevCoord']!=None)):
                        img = ocv.line(img, peC.trackerPool[t]['prevCoord'], peC.trackerPool[t]['currCoord'], (0, 255, 0), 2)
                    if hT.heatChecker==True:
                        minbx = [
                            x-htPointer if (x > htPointer) else 0,
                            b[3]-htPointer if (b[3] > 0) else 0,
                            x+htPointer if (x < (imgShape[1] - htPointer)) else imgShape[1],
                            b[3]+htPointer if (b[3] < (imgShape[0] - htPointer)) else imgShape[0]
                        ]
                        hT.heatMapArr[minbx[1]:minbx[3], minbx[0]:minbx[2]] += 1
                except:
                    st.warning('Bad Frame Intercepted !!!', icon="⚠️")
        if (peC.personChecker==True):
            img = ocv.line(img, (peC.x1, peC.y1), (peC.x2, peC.y2), (0, 255, 255), 1)
            img = ocv.circle(img, (peC.x1, peC.y1), 2, (0, 255, 0), 2)
            img = ocv.circle(img, (peC.x2, peC.y2), 2, (0, 255, 0), 2)
        if ((hT.heatChecker==True) and (hT.heatMapArr.max() > 0)):
            htmapO = hT.heatMapArr / hT.heatMapArr.max()
            htmap = ocv.GaussianBlur((htmapO * 255).astype(np.uint8), (9,9), 0)
            htmap = ocv.applyColorMap(htmap, ocv.COLORMAP_JET)
            imgW = ocv.addWeighted(img, 0.7, htmap, 0.3, 0)
            img[:, :, 0] = np.where(htmapO < 0.1, img[:, :, 0], imgW[:, :, 0])
            img[:, :, 1] = np.where(htmapO < 0.1, img[:, :, 1], imgW[:, :, 1])
            img[:, :, 2] = np.where(htmapO < 0.1, img[:, :, 2], imgW[:, :, 2])
            img = img.astype(np.uint8)
    return av.VideoFrame.from_ndarray(img, format="bgr24")

webrtc_streamer(
    key = 'face',
    mode = WebRtcMode.SENDRECV,
    video_frame_callback = frameProcessor,
    # rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints = {'video': True, 'audio': False},
    async_processing = True
)

hide_streamlit_style = """
            <style>
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
