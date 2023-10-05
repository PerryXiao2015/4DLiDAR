from monodepth2  import monodepth2
md = monodepth2()

import cv2

#device_id = 0 #camera_device id 
camera = cv2.VideoCapture(0)
image_size = 480

while camera.isOpened():
    ok, cam_frame = camera.read()
    if not ok:
        break
    
    #cam_frame= cv2.resize(cam_frame, (image_size, image_size))     
    # Load in a frame
    depth = md.eval(cam_frame)
   
    cv2.imshow('video image', cam_frame)
    cv2.imshow('MonoDepth', depth)
    key = cv2.waitKey(30)
    if key == 27: # press 'ESC' to quit
        break

camera.release()