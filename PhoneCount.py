import cv2
import MessagePassing as msg
# from alarm import play_audio as play
# import wins
from deepface import DeepFace as dp 
from ultralytics import YOLO
import numpy as np
import supervision as sv
import liveHeadcountupdate as headcount
# import LiveGraphREDBLUE 
from datetime import date, datetime
import time 
import torch
import streamlit as st
from streamlit_webrtc import webrtc_streamer
import av
from selectpoints import *
from shapely.geometry import *
from Datastore import *

if torch.cuda.is_available():
    torch.cuda.set_device(0)  # Select GPU device 0


time_now = datetime.now()
current_time = time_now.strftime("%H:%M:%S")
today = date.today()
# flag =False
# import tkinter
# import customtkinter as ctk
# from PIL import Image, ImageTk
THRESHOLD=10

START = sv.Point(320,0)
END = sv.Point(320,480)

def stream():
    st.title("Crowd Management Dashboard")

    # Button to start processing
    if st.button("Start Processing"):
        # stop_button=st.button("Stop Processing")
        main()
        
        #stop processing
        # if stop_button:
        #     st.stop()
        
        
# Function to update the displayed image
def update_image(empty_slot, new_image):
    empty_slot.image(new_image, channels="BGR")

# def writevideo()
def alarm(count,flag):
    print("ALARM !")
    msg.pushmsg("Platform No. 6", f"Mob Detected:{count}people")
    # wins.main()
    return False

def point_in_polygon(point, polygon):
    point = Point(point)
    return polygon.contains(point)

def get_name(img,x1,y1,x2,y2):
    # print(type(x1),type(y1),type(x2),type(y2))
    x1=int(x1)
    x2=int(x2)
    y1=int(y1)
    y2=int(y2)
    det_face = img[y1:y2,x1:x2] #crop image to face region
    
    try:
        dfs=dp.find(det_face,"Deepface",
                        # model_name="GhostFaceNet",
                        # detector_backend="retinaface",
                        silent= True)
        name = find_name(dfs[0].head(1)['identity'][0])
        print(name)
        cv2.imshow('face',det_face)
        return name
    
    except ValueError as err:
        return 'Unknown'
    
def main():
    # model = YOLO("person_phone_best.pt") // model detecting person as 'Using-wearables' without wearables
    model = YOLO(r"C:\Users\USER\Downloads\last (1).pt")
    # model = YOLO("phone_best.pt")
    # model = YOLO("yolov9c.pt")
    # model = YOLO("E:/OpenCV/best_worker.pt")
    # model= YOLO(r"E:\OpenCV\trashh.pt")
    source="0"
    # source="E:\OpenCV\Crowd_new.mp4"
    # source="http://192.168.0.166:8080/video"
    # area=[(140,80), (520,100), (516,455), (140,440)]
    area=np.array([[140,80], [820,100],[ 816,655], [140,640]])
    polygon = Polygon([[140,80], [820,100],[ 816,655], [140,640]])
    # area=np.array([[140,80], [1000,752],[716,1455], [140,1440]])
    # cv2.namedWindow("Phone Detection")
    # cv2.setMouseCallback("Phone Detection", points)
    
    flag= False
    rec_datetime=""
    
    # model = YOLO("best_head.pt")
    line_zone = sv.LineZone(start=START , end=END)
    line_zone_annotator=  sv.LineZoneAnnotator(
        thickness= 2,
        text_thickness=1,
        text_scale=0.5, 
    )
    
    #initiate annotators
    box_annotator = sv.BoxAnnotator(
        thickness = 2,
        text_thickness = 1,
        text_scale = 0.5
    )
    zone=sv.PolygonZone(
        polygon=area,
        frame_resolution_wh=(480,640))
    zone_annotator=sv.PolygonZoneAnnotator(zone=zone, color=sv.Color.green(), thickness=1,text_scale=0)

    print(model.model.names)
    # stframe = st.empty()
    # print(model.model.names) ****
    # fc=1 #Framecount
    for result in model.track(source=source,show=False, stream=True):
    # for result in model.track(source="0" , show=False, stream=True):
        # stframe = st.empty()
        frame= result.orig_img
        print(frame.shape)
        # print("result: ",result[0])
        detections = sv.Detections.from_ultralytics(result)
        # zone.trigger(detections=detections)
        
        if result.boxes.id is not None:
            detections.tracker_id = result.boxes.id.cpu().numpy().astype(int)

        print(detections)
        
        
        detections = detections[detections.confidence > 0.3] #Confidence threshold
        # detections = detections[detections.class_id == 0 ]++
        
        # labels = [
        #     f"#{tracker_id} {model.model.names[class_id]} {confidence: 0.2f}"
        #     for _,confidence, class_id,tracker_id in detections
        # ]
        # print(detections)
        labels=[]
        
        excluded_detections=[]
        
        # Access the elements of the detections object here
        # print("length:",len(detections.class_id))
        for i in range(len(detections.class_id)):
                            
            top_left = (detections.xyxy[0][0], detections.xyxy[0][1])
            top_right = (detections.xyxy[0][2], detections.xyxy[0][1])
            bottom_left = (detections.xyxy[0][0], detections.xyxy[0][3])
            bottom_right = (detections.xyxy[0][2], detections.xyxy[0][3])

            # Convert bounding box coordinates to polygon
            bbox_polygon = np.array([top_left, top_right, bottom_right, bottom_left]).astype(np.int32)
            
            bbox_center = Point((detections.xyxy[0][0]+detections.xyxy[0][2])/2,(detections.xyxy[0][1]+detections.xyxy[0][3]/2))

            print(f"{detections.xyxy}------------------------->{model.model.names[detections.class_id[i]]}",polygon.contains( bbox_center))
            
            # print(f">>>>>>>>>>>>>>>> coords{}")
            name = get_name(frame,detections.xyxy[0][0],detections.xyxy[0][1],detections.xyxy[0][2],detections.xyxy[0][3]) #Face Recognition module called
            
            # name=""
            # Check if the bounding box is inside the zone
            if polygon.contains( Point(bbox_center) ):
                # Calculate the intersection between the bounding box and the zone
                intersection = polygon.intersection(Polygon(bbox_polygon))
                print("\n_____________",intersection.area)
                # If the intersection is not empty, then the bounding box is inside the zone
                if intersection.area > 0:
                    # print("True")
                    flag = True
                    # labels.append(f"#{detections.tracker_id[i]} {model.model.names[detections.class_id[i]]} {detections.confidence[i]: 0.2f} ")
                    labels.append(f"{name} {model.model.names[detections.class_id[i]]}{detections.confidence[i]: 0.2f}")   
            else:
                # Delete the i-th detection from the sv.Detections object
                # detections = np.delete(detections.detections, 0, 0)
                excluded_detections.append(i)
            print("**************************************",excluded_detections)
            
        # for i in range(len(excluded_detections)-1,0,-1):
        #     detections.xyxy = np.delete(detections.xyxy, i)
        #     detections.confidence = np.delete(detections.confidence, i)
        #     detections.class_id = np.delete(detections.class_id, i)
        #     print("Excluded....")
        
        # print(detections.xyxy[0])
        detections.xyxy=np.delete(detections.xyxy,excluded_detections,0)
        detections.confidence=np.delete(detections.confidence,excluded_detections)
        detections.class_id=np.delete(detections.class_id,excluded_detections)
        print("Excluded....")
        print(detections)
        # labels=[f"{model.model.names[detections.class_id[0]]}"]
        # for _,confidence, class_id,tracker_id in detections:
        #     labels.append(f"#{tracker_id} {model.model.names[class_id]} {confidence: 0.2f}")
        # print(labels[len(labels)-1])
        
        # print(detections.class_id)
        # print("labels:",labels)
        # print("class:",model.model.names[detections[2]])
        frame = box_annotator.annotate(
            scene = frame, 
            detections=detections, 
            labels = labels
        )
        
        # frame = zone_annotator.annotate(scene= frame, label=None)
        # line_zone.trigger(detections)
        # line_zone_annotator.annotate(frame,line_zone)
        
        # Define the coordinates of the top-left and bottom-right corners of the box
        top_left = (20, 20)
        bottom_right = (80, 80)
        
        # Count the occurrence of number of persons in the array
        # count = np.count_nonzero(detections.class_id == 0)
        
        # count = np.count_nonzero(detections.class_id == 5) #for Safety model
        count = np.count_nonzero(detections.class_id) #for Safety model


        # Draw the box
        cv2.rectangle(frame, top_left, bottom_right, (82,127, 212), -1)

        # Write the number inside the box
        cv2.putText(frame, "phone count", (20,20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)#count 
        cv2.putText(frame, f"{count}", (40,60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)#number of occurances
        
        cv2.putText(frame, "1", (140,80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        cv2.putText(frame, "2", (820,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        cv2.putText(frame, "3", (816,655), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        cv2.putText(frame, "4", (140,640), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        cv2.polylines(frame,[np.array(area,np.int32)],True, (255,0,0),1)
        
        if (count > THRESHOLD):
            alarm(count,flag)
        # current_time=datetime.datetime.now()
        #if "p" is pressed the alarm will stop 
        
            
        # flag =False
        # alarm(count,flag)
            
            
        #Record the Head count in a csv file
        time_now = datetime.now()
        current_time = time_now.strftime("%H:%M:%S")
        today = date.today()
        
        if not (f"{today} {current_time}"== rec_datetime):
            headcount.recordHead(f"{today} {current_time}", count)
            
        rec_datetime = f"{today} {current_time}"

        # Resize the frame
        # resized_frame = cv2.resize(frame, (720, 480))
        
        #Save video code
        # frame_size = frame.shape
        # # print(frame_size)
        # video.write(frame)
        
        # print(f"frame {fc} written...") #framecount
        # fc+=1
        #Create a VideoWriter object with CIF parameters
        fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Define codec for CIF
        # out = cv2.VideoWriter('output.avi', fourcc, 30, (352, 288))

        # resized_frame = cv2.resize(frame, (720, 480))  # Resize to CIF
        # resized_frame = cv2.resize(frame, (1280,720))  # Resize to CIF
        # out.write(resized_frame)  # Write frame to video
        
        
        
        cv2.imshow("Phone Detection", frame)
    
        
        # if(cv2.waitKey(30)==ord('q')):
        if(cv2.waitKey(1) & 0xFF == ord("q")):
            # cv2.destroyAllWindows()
            break
        
        # Display the frame in Streamlit
        # stframe.image(resized_frame, channels="BGR")

        # # Clear the previous frame
        # st.empty()

        # # Add a delay to control the frame rate
        # time.sleep(0.01)
        
        # stframe.image(frame, channels="BGR", use_column_width=True)
        
        # return resized_frame
        # frame_callback(frame)
        # cv2.imshow("yolov8", frame)
        # print("Done")
        # return frame
    # video.release()    

main()
# stream()