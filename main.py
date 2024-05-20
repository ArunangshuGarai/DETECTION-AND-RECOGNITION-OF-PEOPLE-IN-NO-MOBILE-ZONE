import threading
import cv2
import MessagePassing as msg
import Audio_player
# from alarm import play_audio as play
from deepface import DeepFace as dp 
from ultralytics import YOLO 
import numpy as np
import pandas as pd
import supervision as sv
import torch
import streamlit as st
# from selectpoints import *
from shapely.geometry import *
from Datastore import *
import tempfile
import pygame
import random
import datetime
from PIL import Image
from Database import capture_snapshots
from Database import print_database

if torch.cuda.is_available():
    torch.cuda.set_device(0)  # Select GPU device 0

# flag =False

THRESHOLD=10
interval = 100 
START = sv.Point(320,0)
END = sv.Point(320,480)
alarm_interval = 10


def stream():
    
    im = Image.open("favicon.ico")
    st.set_page_config(
        page_title="Phone Count System App",
        page_icon=im,
        layout="wide",
        initial_sidebar_state="expanded"
    )
    st.title("Phone Count System")  
    st.sidebar.title("Settings")
    src = st.sidebar.radio(
        "Choose one of the sources below",
        ["Laptop Camera","IP Camera", "Media upload"],
        captions=["","http://192.168.3.127:8080/video",""]
    )
    
    if src == "Laptop Camera":
        source=0
    elif src == "IP Camera":
        source = "http://192.168.3.127:8080/video"
    elif src == "Media upload":
        video_buffer = st.sidebar.file_uploader("Choose a video", type=["mp4" , "avi" , "mov" , "asf", "m4v",'jpg', "jpeg", "png"])
        DEMO_VIDEO = 'my_video.mp4'
        tffile = tempfile.NamedTemporaryFile(suffix = '.mp4', delete=False)
        
        
                
        if not video_buffer:
            vid = cv2.VideoCapture(DEMO_VIDEO)
            tffile.name = DEMO_VIDEO
            dem_vid = open(tffile.name, 'rb') 
            demo_bytes = dem_vid.read()
            source = tffile.name
            
            # st.sidebar.text('Input Video')
            # st.sidebar.video(demo_bytes)
        
        else:
            tffile.write(video_buffer.read())
            dem_vid = open(tffile.name, 'rb') 
            demo_bytes = dem_vid.read()
            source = tffile.name
            
            st.sidebar.text('Input Video')
            st.sidebar.video(demo_bytes)
            
    elif src == "YouTube video":
        source = st.sidebar.text_input("Enter YouTube Link")
    st.sidebar.markdown("---")    
    
    col1, col2, col3 = st.columns([5,2,2])

    # snap_placeholder = st.empty()
    
    with col1:
        frame_placeholder=st.empty()

    with col2:

            snap_placeholder = st.empty()
    
    with col3:
        
            info_placeholder = st.empty()
        
            
    
    if (st.button('Start')):

            stop_button_pressed= st.button('Stop')

            main(stop_button_pressed, frame_placeholder,snap_placeholder,info_placeholder, source)
    
    # checkbox = st.sidebar.checkbox("Show the detected labels", value = True)
    
    
        
        
# Function to update the displayed image
def update_image(empty_slot, new_image):
    empty_slot.image(new_image, channels="BGR")

# def writevideo()
def generate_alarm():
    n=random.randint(1,3)
    if n==1:
        audio_file_path = r"SBH_R_ENGLISH.mp3"
    elif n==2:
        audio_file_path = r"SBH_R_BENGALI.mp3" 
    else:
        audio_file_path = r"SBH_R_HINDI.mp3"
    duration = 8
    Audio_player.play_audio(audio_file_path)

# def point_in_polygon(point, polygon):
#     point = Point(point)
#     return polygon.contains(point)

def get_name(img,x1,y1,x2,y2,zone):
    # print(type(x1),type(y1),type(x2),type(y2))
    global alarm_interval
    name="Unknown"
    
    x1=int(x1)
    x2=int(x2)
    y1=int(y1)
    y2=int(y2)
    det_face = img[y1:y2,x1:x2] #crop image to face region
    
    if alarm_interval == 0:
        p1 = threading.Thread(target = generate_alarm)
        p1.start()
        alarm_interval=1000
        print(f"alarm_interval={alarm_interval}")

    
    try:
        dfs=dp.find(det_face,"Deepface",
                        # model_name="GhostFaceNet",
                        # detector_backend="retinaface",
                        silent= True)
        if len(dfs[0]) > 0: name = find_name(dfs[0].head(1)['identity'][0]) 
        print(name)
        # p2 = threading.Thread(target = msg.pushmsg, args = (zone, name))
        # p2.start()
        # cv2.imshow('face',det_face)
        return name , det_face
    
    except ValueError as err:
        return 'Unknown' , det_face
    
def updateIntervals(interval_list): #Snapshot interval update 
    for i in list(interval_list.keys()):
            print (interval_list[i],end=" ")
            interval_list[i]-= 1
            
            if (interval_list[i]==0):
                del interval_list[i]
    print(interval_list)       
    return interval_list

   

def main(stop_button_pressed, frame_placeholder,snap_placeholder, info_placeholder, source=0):
# def main():
    global alarm_interval
    alarm_interval=10
    # model = YOLO("C:\Users\USER\Downloads\best150n.pt") // model detecting person as 'Using-wearables' without wearables
    # model = YOLO(r"phone_best.pt")
    model = YOLO(r"C:\Users\USER\Downloads\best150n.pt") 
    # model = YOLO("yolov9c.pt")
    # model = YOLO("E:/OpenCV/best_worker.pt")
    # model= YOLO(r"E:\OpenCV\trashh.pt")
    # source="0"
    # source="Walking While Texting _ Crowd Control.mp4"
    # source="http://192.168.3.127:8080/video"
    
    # area=[(140,80), (520,100), (516,455), (140,440)]
    area1=np.array([[70,70], [550,70],[550,675], [70,675]])
    polygon1 = Polygon([[70,70], [550,70],[550,675], [70,675]])
    area2=np.array([[745,50], [1220,50],[1220,664], [745,664]])
    polygon2 = Polygon([[745,50], [1220,50],[1220,664], [745,664]])
    # area=np.array([[140,80], [820,100],[ 816,655], [140,640]])
    # polygon = Polygon([[140,80], [820,100],[ 816,655], [140,640]])
    # area=np.array([[140,80], [1000,752],[716,1455], [140,1440]])
    # cv2.namedWindow("Phone Detection")
    # cv2.setMouseCallback("Phone Detection", points)
    
    flag= False
    rec_datetime=""

    #initiate annotators
    box_annotator = sv.BoxAnnotator(
        thickness = 2,
        text_thickness = 1,
        text_scale = 0.5
    )
    zone=sv.PolygonZone(
        polygon=area1,
        frame_resolution_wh=(480,640))
    zone_annotator=sv.PolygonZoneAnnotator(zone=zone, color=sv.Color.green(), thickness=1,text_scale=0)

    print(model.model.names)
    
    frame_count=0 #Framecount
    int_list={}
    
    db_placeholder = st.empty()
    button_clicked = st.button("Database")
    
    if button_clicked:
            df = pd.read_csv('snapshot_data.csv')
            db_placeholder.dataframe(df)
    
    for result in model.track(source=source,show=False, stream=True):
    # for result in model.track(source="0" , show=False, stream=True):
        # stframe = st.empty()
        frame= result.orig_img
        
        # frame = cv2.resize(frame, (1280,720))

        frame_count+=1 # Frame count update
        
        # print(alarm_interval)
        if alarm_interval > 0:
            alarm_interval = alarm_interval -1 
        
        print(f"alarm_interval={alarm_interval}")
        
        int_list=updateIntervals(int_list) # Decease snapshot intervals of each person in the List
        
        
        print(frame.shape)
        # print("result: ",result[0])
        detections = sv.Detections.from_ultralytics(result)
        # zone.trigger(detections=detections)
        
        if result.boxes.id is not None:
            detections.tracker_id = result.boxes.id.cpu().numpy().astype(int)

        print(detections)
        
        
        detections = detections[detections.confidence > 0.3] #Confidence threshold
        # detections = detections[detections.class_id != 3 ]
        
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

            print(f"{detections.xyxy}------------------------->{model.model.names[detections.class_id[i]]}",polygon1.contains(bbox_center))
            
            # print(f">>>>>>>>>>>>>>>> coords{}")
            
            # Check if the bounding box is inside the zone
            if polygon1.contains( Point(bbox_center) ) or polygon2.contains( Point(bbox_center) ) :
                
                # Determine zone                
                if polygon1.contains( Point(bbox_center) ):
                    zone = 1
                elif polygon2.contains( Point(bbox_center) ):
                    zone = 2
                
                current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                name , snap = get_name(frame,detections.xyxy[0][0],detections.xyxy[0][1],detections.xyxy[0][2],detections.xyxy[0][3], zone) #Face Recognition module called
                
                # name=""
                
                if (name not in int_list.keys() ):
                    
                    if name.lower() == 'unknown':
                        int_list[name] = interval//2
                    else:
                        int_list[name] = interval
                    
                    snap = cv2.resize(snap, (360,640))
                    # Write the number inside the box
                    cv2.putText(snap, f"{name}", (1,snap.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
                    
                    p2 = threading.Thread(target = msg.pushmsg, args = (zone, name))
                    p2.start()
                    
                    # cv2.imshow('face',snap)
                    snap_placeholder.image(snap, channels="BGR")
                    
                    print(f"{int_list} -----------> {name} captured")
                    #snapshot(snap) # ! Save snapshot
                    output_dir = r"Snapshots"  # Output directory to save the snapshots
                    # filepath = os.path.join(output_dir, f"{current_time}{name}")
                    # cv2.imwrite(filepath, snap)
                    saved_path = capture_snapshots.capture_snapshots(snap, output_dir, current_time, name , zone)
                    print_database.export_database_to_csv()#db() # ! connect to Db
                    print("Snapped....")
                    
                    # info_placeholder.write(name, current_time , f"Zone{zone}", f"Saved to {saved_path}")
                    # info_placeholder.write(name)
                    # info_placeholder.write(current_time)
                    # info_placeholder.write(f"Zone{zone}")
                    # info_placeholder.write(f"Saved to {saved_path}")
                    # Replace the chart with several elements:
                    with info_placeholder.container():
                        st.write("Name : ", name)
                        st.write("Timstamp: ",current_time)
                        st.write(f"Platform {zone}")
                        st.write(f"Saved to: {saved_path}")
                    
                
                #interval_list['SB'] = 100
                
                # Calculate the intersection between the bounding box and the zone
                intersection1 = polygon1.intersection(Polygon(bbox_polygon))
                intersection2 = polygon2.intersection(Polygon(bbox_polygon))
                print("\n_____________",intersection1.area)
                # If the intersection is not empty, then the bounding box is inside the zone
                if intersection1.area > 0 or intersection2.area > 0 :
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
        # print(detections)
     
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
        
        cv2.putText(frame, "Zone 1", (75,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (200,0,0), 2)
        cv2.putText(frame, "Zone 2", (750,80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,200), 2)
        # cv2.putText(frame, "2", (820,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        # cv2.putText(frame, "3", (816,655), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        # cv2.putText(frame, "4", (140,640), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        # frame = cv2.resize(frame, (1280,720))

        cv2.polylines(frame,[np.array(area1,np.int32)],True, (255,0,0),1)
        cv2.polylines(frame,[np.array(area2,np.int32)],True, (0,0,255),1)
        

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
        
        
        # cv2.imshow("Phone Detection", frame)
    
        # frame = cv2.resize(frame, (1280,720))

        # if(cv2.waitKey(30)==ord('q')):
        if(cv2.waitKey(1) & 0xFF == ord("q") ):
            # cv2.destroyAllWindows()
            break
        
        # Display the frame in Streamlit
        frame_placeholder.image(frame, channels="BGR")
        # if button_clicked:
        df = pd.read_csv('snapshot_data.csv')
        # db_placeholder.table(df)
        db_placeholder.table(df)

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

# main()
stream()