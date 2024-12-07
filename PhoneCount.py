import threading
import cv2
import MessagePassing as msg
import Audio_player
from deepface import DeepFace as dp 
from ultralytics import YOLO 
import numpy as np
import pandas as pd
import supervision as sv
import torch
import streamlit as st
from shapely.geometry import *
from Datastore import *
import tempfile
import random
import datetime
from PIL import Image
from Database import capture_snapshots_DB
from Database import print_database

if torch.cuda.is_available():
    torch.cuda.set_device(0)  # Select GPU device 0



THRESHOLD=10
interval = 100 
START = sv.Point(320,0)
END = sv.Point(320,480)
alarm_interval = 20


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
        captions=["","",""]
    )
    
    if src == "Laptop Camera":
        source=0
    elif src == "IP Camera":
        # source = "http://192.168.3.127:8080/video"
        source = st.sidebar.text_input("IP camera","http://192.168.0.100:8080/video")
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
        
        return name , det_face
    
    except ValueError as err:
        return 'Unknown' , det_face
    
def updateIntervals(interval_list): #Snapshot interval update 
    for i in list(interval_list.keys()):
            # print (interval_list[i],end=" ")
            interval_list[i]-= 1
            
            if (interval_list[i]==0):
                del interval_list[i]
    print(interval_list)       
    return interval_list

   

def main(stop_button_pressed, frame_placeholder,snap_placeholder, info_placeholder, source=0):
    global alarm_interval
    alarm_interval=10
    
    #Load the model
    # model = YOLO(r"models\phone_best.pt")
    model = YOLO("models\best150n.pt") # model detecting person as 'Using-wearables' without wearables
    # model = YOLO(r"C:\Users\USER\Downloads\best150n.pt")
    # model = YOLO("E:/OpenCV/best_worker.pt")
    

    area1=np.array([[70,70], [550,70],[550,675], [70,675]])
    polygon1 = Polygon([[70,70], [550,70],[550,675], [70,675]])
    area2=np.array([[745,50], [1220,50],[1220,664], [745,664]])
    polygon2 = Polygon([[745,50], [1220,50],[1220,664], [745,664]])
    
    
    flag= False
    rec_datetime=""

    #initiate annotators
    box_annotator = sv.BoxAnnotator(
        thickness = 2
    )
    
    label_annotator = sv.LabelAnnotator(
        # thickness = 2,
        text_thickness = 1,
        text_scale = 0.5,
        text_padding= 1,
        text_position=sv.Position.TOP_LEFT
        )
    zone=sv.PolygonZone(
        polygon=area1,
        frame_resolution_wh=(480,640))
    # zone_annotator=sv.PolygonZoneAnnotator(zone=zone, color=sv.Color.green(), thickness=1,text_scale=0)

    print(model.model.names)
    
    frame_count=0 #Framecount
    int_list={}
    
    db_placeholder = st.empty()
    button_clicked = st.button("Database")
    
    if button_clicked:
            df = pd.read_csv('snapshot_data.csv')
            db_placeholder.dataframe(df)
    
    for result in model.track(source=source,show=False, stream=True):
        
        frame= result.orig_img
        
        frame_count+=1 # Frame count update
        
        # print(alarm_interval)
        if alarm_interval > 0:
            alarm_interval = alarm_interval -1 
        
        print(f"alarm_interval={alarm_interval}")
        
        int_list=updateIntervals(int_list) # Decease snapshot intervals of each person in the List
        
        
        print(frame.shape)
    
        detections = sv.Detections.from_ultralytics(result)
        
        
        if result.boxes.id is not None:
            detections.tracker_id = result.boxes.id.cpu().numpy().astype(int)

        
        
        
        detections = detections[detections.confidence > 0.3] #Confidence threshold
        # detections = detections[detections.class_id != 3 ]
        
        labels=[]
        
        excluded_detections=[]
        
        # Access the elements of the detections object here
        for i in range(len(detections.class_id)):
                          
            top_left = (detections.xyxy[0][0], detections.xyxy[0][1])
            top_right = (detections.xyxy[0][2], detections.xyxy[0][1])
            bottom_left = (detections.xyxy[0][0], detections.xyxy[0][3])
            bottom_right = (detections.xyxy[0][2], detections.xyxy[0][3])

            # Convert bounding box coordinates to polygon
            bbox_polygon = np.array([top_left, top_right, bottom_right, bottom_left]).astype(np.int32)
            
            bbox_center = Point((detections.xyxy[0][0]+detections.xyxy[0][2])/2,(detections.xyxy[0][1]+detections.xyxy[0][3]/2))

            print(f"{detections.xyxy}------------------------->{model.model.names[detections.class_id[i]]}",polygon1.contains(bbox_center))
            
            
            # Check if the bounding box is inside the zone
            if polygon1.contains( Point(bbox_center) ) or polygon2.contains( Point(bbox_center) ) :
                
                # Determine zone                
                if polygon1.contains( Point(bbox_center) ):
                    zone = 1
                elif polygon2.contains( Point(bbox_center) ):
                    zone = 2
                
                current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                #Face Recognition module called
                name , snap = get_name(frame,detections.xyxy[0][0],detections.xyxy[0][1],detections.xyxy[0][2],detections.xyxy[0][3], zone) 
                
                
                # if name is not found in the interval list , then alarm & msg will be triggered
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
                    
                    snap_placeholder.image(snap, channels="BGR")
                    
                    print(f"{int_list} -----------> {name} captured")
                    
                    # Save snapshot
                    output_dir = r"Snapshots"  # Output directory to save the snapshots
                    
                    saved_path = capture_snapshots_DB.capture_snapshots(snap, output_dir, current_time, name , zone)
                    
                    # connect to Db
                    print_database.export_database_to_csv()#db() 
                    print("Snapped....")
                    
        
                    with info_placeholder.container():
                        st.write("Name : ", name)
                        st.write(model.model.names[detections.class_id[i]])
                        st.write("Timstamp: ",current_time)
                        st.write(f"Platform {zone}")
                        st.write(f"Saved to: {saved_path}")
                    
                
                # Calculate the intersection between the bounding box and the zone
                intersection1 = polygon1.intersection(Polygon(bbox_polygon))
                intersection2 = polygon2.intersection(Polygon(bbox_polygon))
                print("\n_____________",intersection1.area)
                
                # If the intersection is not empty, then the bounding box is inside the zone
                if intersection1.area > 0 or intersection2.area > 0 :
                                      
                    labels.append(f"{name} {model.model.names[detections.class_id[i]]}{detections.confidence[i]: 0.2f}")   
            else:
                # Delete the i-th detection from the sv.Detections object                
                excluded_detections.append(i)            

        
        #Excluding (Deleting) detections outside the Zones
        detections.xyxy=np.delete(detections.xyxy,excluded_detections,0)
        detections.confidence=np.delete(detections.confidence,excluded_detections)
        detections.class_id=np.delete(detections.class_id,excluded_detections)
        print("Excluded detections....")
        
        #Draw annotations on the frame (bboxes and labels)
        frame = box_annotator.annotate(
            scene = frame, 
            detections=detections
        )
        frame = label_annotator.annotate(
            scene = frame, 
            detections=detections, 
            labels = labels
        )
        
        
        # Define the coordinates of the top-left and bottom-right corners of the box
        top_left = (20, 20)
        bottom_right = (80, 80)
        
        count = np.count_nonzero(detections.class_id) #Count the no of devices detected

        # Draw the box around "Phone count"
        cv2.rectangle(frame, top_left, bottom_right, (82,127, 212), -1)

        # Write the number inside the box
        cv2.putText(frame, "phone count", (20,20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)#count 
        cv2.putText(frame, f"{count}", (40,60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)#number of occurances
        
        cv2.putText(frame, "Zone 1", (75,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (200,0,0), 2)
        cv2.putText(frame, "Zone 2", (750,80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,200), 2)
        
        #Drawing the Zones
        cv2.polylines(frame,[np.array(area1,np.int32)],True, (255,0,0),1)
        cv2.polylines(frame,[np.array(area2,np.int32)],True, (0,0,255),1)
        
        
        if(cv2.waitKey(1) & 0xFF == ord("q") ):
            cv2.destroyAllWindows()
            break
        
        # Display the frame in Streamlit
        frame_placeholder.image(frame, channels="BGR")
        
        # if button_clicked:
        df = pd.read_csv('snapshot_data.csv')
        
        db_placeholder.table(df)
  
# main()
stream()