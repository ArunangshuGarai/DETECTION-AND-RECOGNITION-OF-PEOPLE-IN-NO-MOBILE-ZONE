import cv2
import numpy as np
from ultralytics import YOLO

def points(event , x, y, flags, param):
    print(event)
    if event== cv2.EVENT_MOUSEMOVE :
        colorsBGR = [x,y]
        print(colorsBGR)

def video():        
    cv2.namedWindow("ROI")
    cv2.setMouseCallback("ROI", points)

    model = YOLO("yolov8n.pt")
    source=r"Test Media\SIH.jpg" 

    for result in model.track(source=source,show=False, stream=True):

        # stframe = st.empty()
        frame= result.orig_img

        # cv2.imshow("ROI", frame)
        
        resized_frame = cv2.resize(frame, (1280,720))  # Resize to CIF
        # out.write(resized_frame)  # Write frame to video
            
        cv2.imshow("ROI", resized_frame)
        
            
        # if(cv2.waitKey(30)==ord('q')):
        if(cv2.waitKey(0) & 0xFF == ord("q")):
            # cv2.destroyAllWindows()
            break
video()