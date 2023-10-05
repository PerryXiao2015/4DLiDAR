# =================================================================================================#
# 4DLiDAR Demo Programme
# Professor Perry Xiao
# London South Bank University
# Date: October 05, 2023
# =================================================================================================#
#
# https://google.github.io/mediapipe/solutions/solutions.html

#pip install --upgrade protobuf
#pip install streamlit

import streamlit as st
import cv2 
import numpy as np
import pandas as pd
import time
from datetime import datetime
import matplotlib.pyplot as plt

# imports for reproducibility
import tensorflow as tf
import random
import os
from keras import backend as K

# MonoDepth2 Library ==============================================================================#
from monodepth2  import monodepth2
md = monodepth2()

# Mediapipe library for Face Mesh, Hand, Pose and Holistic Detection ==============================#
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(
    #enable_segmentation=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

#mp_drawing = mp.solutions.drawing_utils
#mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

#mp_drawing = mp.solutions.drawing_utils
#mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

#mp_drawing = mp.solutions.drawing_utils
#mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    #model_complexity=1,
    #smooth_landmarks=True,
    #refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

    
# ----- Define the Webcam functions ---------------------------------------------------------------#
#@st.cache(allow_output_mutation=True)
@st.cache_resource()
def get_cap(id):
    return cv2.VideoCapture(id)

# ----- Define the getHolistic functions ----------------------------------------------------------#
# https://google.github.io/mediapipe/solutions/holistic.html
#@st.cache(allow_output_mutation=True)
@st.cache_resource()
def getHolistic(image):
        # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = holistic.process(image)

    # Draw landmark annotation on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    #Draw face landmarks
    mp_drawing.draw_landmarks(
        image,
        results.face_landmarks,
        mp_holistic.FACEMESH_CONTOURS,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp_drawing_styles
        .get_default_face_mesh_contours_style())
    # Draw Pose landmarks
    mp_drawing.draw_landmarks(
        image,
        results.pose_landmarks,
        mp_holistic.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles
        .get_default_pose_landmarks_style())
    # Draw hand landmarks
    # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                             #mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
                             #mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                             mp_drawing.DrawingSpec(color=(255,0,0), thickness=2, circle_radius=4),
                             mp_drawing.DrawingSpec(color=(255,0,0), thickness=2, circle_radius=2)
                             )
    # Draw right hand connections 
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                             #mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                             #mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                             mp_drawing.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=4),
                             mp_drawing.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=2)
                             )
    

    if results.pose_landmarks:
      image_height, image_width, _ = image.shape
      print(
          f'Nose coordinates: ('
          f'{results.pose_landmarks.landmark[mp_holistic.PoseLandmark.NOSE].x * image_width}, '
          f'{results.pose_landmarks.landmark[mp_holistic.PoseLandmark.NOSE].y * image_height})'
      )
      center_coordinates = (int(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.NOSE].x * image_width),int(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.NOSE].y * image_height))
      image = cv2.circle(image, center_coordinates, 5, (0,0,0), 2)
      cv2.putText(image, 'Nose', center_coordinates, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    if results.left_hand_landmarks:
      print(
          f'Left Index Finger coordinates: ('
          f'{results.left_hand_landmarks.landmark[mp_holistic.HandLandmark.INDEX_FINGER_TIP].x * image_width}, '
          f'{results.left_hand_landmarks.landmark[mp_holistic.HandLandmark.INDEX_FINGER_TIP].y * image_height})'
      )
      center_coordinates = (int(results.left_hand_landmarks.landmark[mp_holistic.HandLandmark.INDEX_FINGER_TIP].x * image_width),int(results.left_hand_landmarks.landmark[mp_holistic.HandLandmark.INDEX_FINGER_TIP].y * image_height))
      image = cv2.circle(image, center_coordinates, 10,  (255, 255, 255), 2)
      cv2.putText(image, 'Left', center_coordinates, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    if results.right_hand_landmarks:
      print(
          f'Right Index Finger coordinates: ('
          f'{results.right_hand_landmarks.landmark[mp_holistic.HandLandmark.INDEX_FINGER_TIP].x * image_width}, '
          f'{results.right_hand_landmarks.landmark[mp_holistic.HandLandmark.INDEX_FINGER_TIP].y * image_height})'
      )
      center_coordinates = (int(results.right_hand_landmarks.landmark[mp_holistic.HandLandmark.INDEX_FINGER_TIP].x * image_width),int(results.right_hand_landmarks.landmark[mp_holistic.HandLandmark.INDEX_FINGER_TIP].y * image_height))
      image = cv2.circle(image, center_coordinates, 10,  (255, 255, 255), 2)
      cv2.putText(image, 'Right', center_coordinates, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
      #print(results.pose_landmarks)
    return image

# ----- Define the getFaceMesh functions ----------------------------------------------------------#
# https://google.github.io/mediapipe/solutions/face_mesh.html
#@st.cache(allow_output_mutation=True)
@st.cache_resource()
def getFaceMesh(image):
    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image)

    # Draw the face mesh annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.multi_face_landmarks:
      for face_landmarks in results.multi_face_landmarks:
        mp_drawing.draw_landmarks(
            image=image,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles
            .get_default_face_mesh_tesselation_style())
        mp_drawing.draw_landmarks(
            image=image,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles
            .get_default_face_mesh_contours_style())
        mp_drawing.draw_landmarks(
            image=image,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_IRISES,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles
            .get_default_face_mesh_iris_connections_style())
        #print(face_landmarks)
      #print(results.multi_face_landmarks)
    return image

# ----- Define the get_label functions ------------------------------------------------------------#
# https://morioh.com/p/afd933f6f3d9
# https://github.com/nicknochnack/AdvancedHandPoseWithMediaPipe/blob/main/Advanced%20HandPose%20Tracking.ipynb
# To get the label of hand
def get_label(image, index, hand, results):
    image_height, image_width, _ = image.shape
    output = None
    for idx, classification in enumerate(results.multi_handedness):
        if classification.classification[0].index == index:
            
            # Process results
            label = classification.classification[0].label
            score = classification.classification[0].score
            text = '{} {}'.format(label, round(score, 2))
            
            # Extract Coordinates
            coords = tuple(np.multiply(
                np.array((hand.landmark[mp_hands.HandLandmark.WRIST].x, hand.landmark[mp_hands.HandLandmark.WRIST].y)),
            [image_width,image_height]).astype(int))
            
            output = text, coords
            
    return output
# ----- Define the draw_finger_angles functions ---------------------------------------------------#

def draw_finger_angles(image, results, joint_list):
    image_height, image_width, _ = image.shape
    # Loop through hands
    for hand in results.multi_hand_landmarks:
        #Loop through joint sets 
        for joint in joint_list:
            a = np.array([hand.landmark[joint[0]].x, hand.landmark[joint[0]].y]) # First coord
            b = np.array([hand.landmark[joint[1]].x, hand.landmark[joint[1]].y]) # Second coord
            c = np.array([hand.landmark[joint[2]].x, hand.landmark[joint[2]].y]) # Third coord
            
            radians = np.arctan2(c[1] - b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
            angle = np.abs(radians*180.0/np.pi)
            
            if angle > 180.0:
                angle = 360-angle
                 
            #cv2.putText(image, str(round(angle, 2)), tuple(np.multiply(b, [image_width,image_height]).astype(int)),
            #           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(image, str(int(angle)), tuple(np.multiply(b, [image_width,image_height]).astype(int)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    return image
# ----- Define the count_finger functions ---------------------------------------------------------#
# https://bleedai.com/real-time-fingers-counter-hand-gesture-recognizer-with-mediapipe-and-python-2/


def count_finger(image, results):
    countL = 0
    countR = 0
    # Retrieve the height and width of the sample image.
    image_height, image_width, _ = image.shape

    if results.left_hand_landmarks:
        #center_coordinates = (int(results.left_hand_landmarks.landmark[mp_holistic.HandLandmark.INDEX_FINGER_TIP].x * image_width),int(results.left_hand_landmarks.landmark[mp_holistic.HandLandmark.INDEX_FINGER_TIP].y * image_height))
        #image = cv2.circle(image, center_coordinates, 10,  (255, 255, 255), 2)
        #cv2.putText(image, 'Left', center_coordinates, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        if results.left_hand_landmarks.landmark[4].x > results.left_hand_landmarks.landmark[3].x:
            countL = countL + 1
        if results.left_hand_landmarks.landmark[8].y < results.left_hand_landmarks.landmark[6].y:
            countL = countL + 1
        if results.left_hand_landmarks.landmark[12].y < results.left_hand_landmarks.landmark[10].y:
            countL = countL + 1
        if results.left_hand_landmarks.landmark[16].y < results.left_hand_landmarks.landmark[14].y:
            countL = countL + 1
        if results.left_hand_landmarks.landmark[20].y < results.left_hand_landmarks.landmark[18].y:
            countL = countL + 1

    if results.right_hand_landmarks:
        #center_coordinates = (int(results.right_hand_landmarks.landmark[mp_holistic.HandLandmark.INDEX_FINGER_TIP].x * image_width),int(results.right_hand_landmarks.landmark[mp_holistic.HandLandmark.INDEX_FINGER_TIP].y * image_height))
        #image = cv2.circle(image, center_coordinates, 10,  (255, 255, 255), 2)
        #cv2.putText(image, 'Right', center_coordinates, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        if results.right_hand_landmarks.landmark[4].x < results.right_hand_landmarks.landmark[3].x:
            countR = countR + 1
        if results.right_hand_landmarks.landmark[8].y < results.right_hand_landmarks.landmark[6].y:
            countR = countR + 1
        if results.right_hand_landmarks.landmark[12].y < results.right_hand_landmarks.landmark[10].y:
            countR = countR + 1
        if results.right_hand_landmarks.landmark[16].y < results.right_hand_landmarks.landmark[14].y:
            countR = countR + 1
        if results.right_hand_landmarks.landmark[20].y < results.right_hand_landmarks.landmark[18].y:
            countR = countR + 1
    s = "Left: "+ str(countL)
    cv2.putText(image, s, (30,60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA)
    s = "Right: "+ str(countR)
    cv2.putText(image, s, ((image_width - 200),60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA)
    return image
            

# ----- Define the count_finger functions ---------------------------------------------------------#
# https://github.com/pdhruv93/computer-vision/blob/main/fingers-count/fingers-count.py

def count_finger2(image, results):
    image_height, image_width, _ = image.shape
    countL = 0
    countR = 0
    # Loop through hands
    count = 0
    for hand in results.multi_hand_landmarks:
        print(count)
        
        count = count + 1
        label = results.multi_handedness[0].classification[0].label
        print(label)
        if label == "Left":
            if hand.landmark[4].x > hand.landmark[3].x:
                countL = countL + 1
            if hand.landmark[8].y < hand.landmark[6].y:
                countL = countL + 1
            if hand.landmark[12].y < hand.landmark[10].y:
                countL = countL + 1
            if hand.landmark[16].y < hand.landmark[14].y:
                countL = countL + 1
            if hand.landmark[20].y < hand.landmark[18].y:
                countL = countL + 1
        if label == "Right":
            if hand.landmark[4].x < hand.landmark[3].x:
                countR = countR + 1
            if hand.landmark[8].y < hand.landmark[6].y:
                countR = countR + 1
            if hand.landmark[12].y < hand.landmark[10].y:
                countR = countR + 1
            if hand.landmark[16].y < hand.landmark[14].y:
                countR = countR + 1
            if hand.landmark[20].y < hand.landmark[18].y:
                countR = countR + 1
           
    s = "Left: "+ str(countL)
    cv2.putText(image, s, (30,60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA)
    s = "Right: "+ str(countR)
    cv2.putText(image, s, ((image_width - 200),60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA)
    return image

# ----- Define the getHandh functions -------------------------------------------------------------#
# https://google.github.io/mediapipe/solutions/hands.html

#@st.cache(allow_output_mutation=True)
@st.cache_resource()
def getHand(image):
    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image)

    # Draw the hand annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    # Rendering results
    if results.multi_hand_landmarks:
        for num, hand in enumerate(results.multi_hand_landmarks):
            mp_drawing.draw_landmarks(image, hand, mp_hands.HAND_CONNECTIONS, 
                                    mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                                    mp_drawing.DrawingSpec(color=(250, 44, 250), thickness=2, circle_radius=2),
                                     )
            
            # Render left or right detection
            if get_label(image, num, hand, results):
                text, coord = get_label(image,num, hand, results)
                cv2.putText(image, text, coord, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)


            # Draw the index fingers
            image_height, image_width, _ = image.shape
            # Center coordinates
            x = hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width
            y = hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_height
            center_coordinates = (int(x),int(y))
                 
            # Radius of circle
            radius = 20
                  
            # Blue color in BGR
            color = (255, 0, 0)
                  
            # Line thickness of 2 px
            thickness = 2
                  
            # Using cv2.circle() method
            # Draw a circle with blue line borders of thickness of 2 px
            image = cv2.circle(image, center_coordinates, radius, color, thickness)

        # Draw angles to image from joint list
        joint_list = [[8,7,6], [12,11,10], [16,15,14], [20,19,18]]
        draw_finger_angles(image, results, joint_list)
        # count fingers
        #count_finger(image, results)
            
##    if results.multi_hand_landmarks:
##       image_height, image_width, _ = image.shape
##       for hand_landmarks in results.multi_hand_landmarks:
##         mp_drawing.draw_landmarks(
##            image,
##            hand_landmarks,
##            mp_hands.HAND_CONNECTIONS,
##            mp_drawing_styles.get_default_hand_landmarks_style(),
##            mp_drawing_styles.get_default_hand_connections_style())
##         print('hand_landmarks:', hand_landmarks)
##         print(
##              f'Index finger tip coordinates: (',
##              f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width}, '
##              f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_height})'
##         )
##         
##         # Center coordinates
##         x = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width
##         y = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_height
##         center_coordinates = (int(x),int(y))
##             
##         # Radius of circle
##         radius = 20
##              
##         # Blue color in BGR
##         color = (255, 0, 0)
##              
##         # Line thickness of 2 px
##         thickness = 2
##              
##         # Using cv2.circle() method
##         # Draw a circle with blue line borders of thickness of 2 px
##         image = cv2.circle(image, center_coordinates, radius, color, thickness)
       #print(results.multi_hand_landmarks)
    return image


# ----- Define the getPoseh functions -------------------------------------------------------------#
# https://google.github.io/mediapipe/solutions/pose.html
#@st.cache(allow_output_mutation=True)
@st.cache_resource()
def getPose(image):
    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image)

    # Draw the pose annotation on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    mp_drawing.draw_landmarks(
        image,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

    if results.pose_world_landmarks:
        #print(results.pose_world_landmarks)
        dt = datetime.now()
        image_height, image_width, _ = image.shape
        print(
          f'Left HIP coordinates: ('
          f'{results.pose_world_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].x * image_width}, '
          f'{results.pose_world_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].y * image_height})'
        )
        
        ax = results.pose_world_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].x * image_width
        ay = results.pose_world_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].y * image_height
        az = results.pose_world_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].z * image_width
        

        bx = results.pose_world_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE].x * image_width
        by = results.pose_world_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE].y * image_height
        bz = results.pose_world_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE].z * image_width

        cx = results.pose_world_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE].x * image_width
        cy = results.pose_world_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE].y * image_height
        cz = results.pose_world_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE].z * image_width
        
        ax = results.pose_world_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP].x * image_width
        ay = results.pose_world_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP].y * image_height
        az = results.pose_world_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP].z * image_width

        bx = results.pose_world_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE].x * image_width
        by = results.pose_world_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE].y * image_height
        bz = results.pose_world_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE].z * image_width

        cx = results.pose_world_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE].x * image_width
        cy = results.pose_world_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE].y * image_height
        cz = results.pose_world_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE].z * image_width
    return image
# ----- Define the getSteroDepthMap functions -------------------------------------------------------------#
def getSteroDepthMap(imgL,imgR):
    imgL_new=cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
    imgR_new=cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)
    stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)
    #stereo = cv2.StereoBM(cv2.STEREO_BM_BASIC_PRESET,ndisparities=16, SADWindowSize=15)

    disparity = stereo.compute(imgL_new,imgR_new)
    #plt.imshow(disparity,'nipy_spectral')
    #plt.show()

    disparity = (disparity - np.amin(disparity))/(np.amax(disparity)-np.amin(disparity))*255
    #print(imgL.shape)
    w,h,_ = imgL.shape
    img = np.zeros([w,h,3])
    img[:,:,0] = np.ones([w,h])*0/255.0
    img[:,:,1] = np.ones([w,h])*0/255.0
    img[:,:,2] = np.ones([w,h])*disparity/255.0    
    return img
# ----- Start the Main Programme ------------------------------------------------------------------#
t0 = time.time()*1000.0
pTime = t0
dt = datetime.now()

cap = get_cap(0)
cap2 = get_cap(1)
mode = 1

st.title("4D LiDAR Demo App")
st.write("**Webcam 1 and 2**")
frameST = st.empty()
st.write("**Stero Depth Map**")
frameST2 = st.empty()
st.write("**MonoDepth2**")
frameST3 = st.empty()
#st.text("Pose Detection")
st.sidebar.markdown("# Object Detection Models")
option = st.sidebar.selectbox(
     'Select a Detection Model:',
     ["Face Mesh","Hand","Pose","Holistic"], index=0)
st.sidebar.write('You selected:', option)
if option == "Face Mesh":
    K.clear_session()
    #st.title("Face Mesh")
    mode = 1
elif option == "Hand":
    K.clear_session()
    #st.title("Hand")
    mode = 2
elif option == "Pose":
    K.clear_session()
    #st.title("Pose")
    mode = 3
    
elif option == "Holistic":
    K.clear_session()
    #st.title("Holistic")
    mode = 4

flipimage = st.sidebar.checkbox('Flip Image',value=True)
#if flipimage:
#     st.write('Great!')

while True:
    ret, frame = cap.read()
    ret2, frame2 = cap2.read()
       
    # Stop the program if reached end of video
    if not ret:
        cv2.waitKey(3000)
        # Release device
        cap.release()
        break
    if not ret2:
        cv2.waitKey(3000)
        # Release device
        cap2.release()
        break
    if flipimage:
        frame = cv2.flip(frame, 1)
        frame2 = cv2.flip(frame2, 1)

    if mode == 1:
        frame = getFaceMesh(frame)
    elif mode == 2:
        frame = getHand(frame)
    elif mode == 3:
        frame = getPose(frame)
    elif mode == 4:
        frame = getHolistic(frame)


    dt = datetime.now()
    cTime = time.time()*1000.0
    dTime = (cTime - t0)
    fps_rate = int(1000.0 /  (cTime - pTime))
    pTime = cTime

    #Print results on screen
    s = "FPS: " + str(fps_rate) + " Now: "+ str(dt)
    cv2.putText(frame, s, (20, 30),
                       cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)   

    # Combine the frames side by side
    combined_frame = cv2.hconcat([frame, frame2])

    img = getSteroDepthMap(frame,frame2)
    depth = md.eval(frame2)
    
    #combined_frame2 = cv2.hconcat([img, depth])
    
    frameST.image(combined_frame, channels="BGR")
    frameST2.image(img, channels="BGR")
    frameST3.image(depth, channels="BGR")
# ----- End of the Main Programme -----------------------------------------------------------------#
