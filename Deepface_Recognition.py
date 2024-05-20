import cv2
# from numba import jit, cuda
import torch
# import matplotlib.pyplot as plt
from deepface import DeepFace as dp
import pandas as pd
from Datastore import find_name
import streamlit as st
import tempfile



def stream():
    st.title("Deep-Face Recognition System")
      
    st.sidebar.title("Settings")
    src = st.sidebar.radio(
        "Choose one of the sources below",
        ["Laptop Camera","IP Camera", "Media upload","YouTube video"],
        captions=["","http://172.16.38.175:8080/video","",""]
    )
    
    if src == "Laptop Camera":
        source=0
    elif src == "IP Camera":
        source = "http://172.16.38.175:8080/video"
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
        
    col1, col2 = st.columns([1,3])

    with col1:
        pass
    with col2:
        pass
    
    st.sidebar.markdown("---")
    
    # checkbox = st.sidebar.checkbox("Show the detected labels", value = True)
    
    if (st.button('Start')):
    
        frame_placeholder=st.empty()

        stop_button_pressed= st.button('Stop')

        main(stop_button_pressed, frame_placeholder,source)
  
def main(stop_button_pressed, frame_placeholder, source=0):
# def main():
    if torch.cuda.is_available():
        torch.cuda.set_device(0)  # Select GPU device 0
        
    print("Starting Face Detection...\nPlease wait.")
    source=source
    print(source)
    # source=0
    # source="http://192.168.0.105:8080/video"
    # source=r"C:\Users\USER\OneDrive\Pictures\Screenshots\Screenshot 2023-11-24 145604.png"

    # pause_button_pressed = st.button('pause')
    # checkbox = st.checkbox("Show the detected labels", value=True)
    
    cap = cv2.VideoCapture(source)  #webcam
    labels_placeholder = st.empty()
    
    fps_placeholder = st.empty()
    
    if cap.isOpened()==False:
        st.write("Starting Face Detection...\nPlease wait.")
    
    while cap.isOpened() and not stop_button_pressed :
        has_frame, img = cap.read()
        if not has_frame:
            # st.write('The video capture has ended.') #no frames are available
            break
        # Resize the clip to CIF resolution (352x288)
        # img = cv2.resize(img, (352, 288))
        print(img.shape)
        
        n = img.shape[0] / 720 # !scale
        # n=1
        print(f"n={n}")
        faces = dp.extract_faces(img,
                                enforce_detection=False,
                                #  detector_backend = "retinaface"                                
                                ) #list of detected faces
        # print(faces)
        names_table= pd.DataFrame()
        confidence_scores=[]
        detected_names=[]
        if (len(faces) >0):
            for i in range (len(faces)):
                # print(faces[i]['facial_area'])
                
                xywh= faces[i]['facial_area']
                
                face_array= faces[i]['face']
                face_array=cv2.cvtColor(face_array, cv2.COLOR_BGR2RGB)
                
                # print("\n&&&&&&&&&&&&&&&&&&&&&&&\n",type(face_array))
                x= xywh['x']
                y= xywh['y']
                w= xywh['w']
                h= xywh['h']
                # det_face = img[y  : y + h , x  : x + w ] #coordinates of one cropped face
                det_face = img[y - h//4  : y + h + h//4 , x - w // 4  : x + w + w//4] #coordinates of one cropped face - margins expanded
                
                # cv2.imshow('cropped', det_face) 
                # cv2.waitKey(0)
                
                
                try:
                    #return a dictionary with the most similar person
                    print(f"face:{i+1}")
                    # cv2.putText(img, f"{i+1}", ( int(xywh['x']-10* n) , int(xywh['y']-10* n )), cv2.FONT_HERSHEY_SIMPLEX, (1*n), (0,0,0), int(2* n))
                    dfs=dp.find(det_face,"Deepface",
                    # model_name="GhostFaceNet",
                    # detector_backend="retinaface",
                    silent= True)    
                    
                    
                    # dfs=dp.find(det_face,"Deepface")
                    # df1=dfs.head()
                    print(f"length:{len(dfs[0])}")
                    
                    
                    #retrieve the 1st data element under 'identity' from the detections Dataframe
                    # print(dfs) 
                    
                    #Find the Name of the detected face from the Database
                    # print(dfs[0].head())
                    if (len(dfs[0])>0):
                        name = find_name(dfs[0].head(1)['identity'][0])  
                        detected_names.append(name)
                        confidence_scores.append(faces[i]['confidence'])
                        
                         
                        # print("Face Found:",name)
                        # print(dfs[0].head(1),name)
                        label=img.copy()
                        # print((xywh['x'],int(xywh['y']-40* n) ))
                        # cv2.rectangle(label,(xywh['x'],int(xywh['y']-40* n) ),(int(xywh['x']+100* n) ,xywh['y']),(52, 177, 235),-1) #bgr
                        cv2.rectangle(label,(xywh['x'],int(xywh['y']-40* n) ),(int(xywh['x']+xywh['w']) ,xywh['y']),(52, 177, 235),-1) #bgr
                
                        alpha = 0.5 #transparency
                        
                        img=cv2.addWeighted(label,alpha, img, 1 - alpha, 0) #add transparent label to image
                        
                        print(name,int(xywh['x']+10* n) , int(xywh['y']-10* n ))
                        cv2.putText(img, f"{name}", ( int(xywh['x']+10* n) , int(xywh['y']-10* n )), cv2.FONT_HERSHEY_SIMPLEX, (1*n), (100,0,1), int(2* n))
                    
                except ValueError as err:
                    dfs = []
                    name = ""
                    print("ValueError")
                    # cv2.putText(img, f"Match error", ( int(xywh['x']+10* n) , int(xywh['y']-10* n )), cv2.FONT_HERSHEY_SIMPLEX, (1*n), (100,0,1), int(2* n))
                    # cv2.putText()
                    # cv2.imshow('ValueError',det_face)
                    # cv2.waitKey(0)
            #   cv2.circle(img,(xywh['x'],xywh['y']), 50, (255,0,255), -1 )
            #   cv2.circle(img,(xywh['x']+xywh['w'],xywh['y']-xywh['h']),50, (255,255,255), -1 )
                cv2.rectangle(img, (xywh['x'],xywh['y']), (xywh['x']+xywh['w'],xywh['y']+xywh['h']), (0,0,220), 2)  #annotate faces
                
                        
            print(f"\n {len(faces)} faces")
            
            if(img.shape[0]> 720): #resolution manipulation
                # img = cv2.resize(img, (540,int(img.shape[0]/img.shape[1]*540)))
                img = cv2.resize(img, (int(img.shape[1]/img.shape[0]*720),720))
        print(img.shape)
        # cv2.imshow("img", img ) #display window outside streamlit

        frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        frame_placeholder.image(frame, channels="RGB")
        
        names_table=pd.DataFrame(list(zip(detected_names, confidence_scores)),
                  columns=['Names', 'Confidence'])
      
        # fps = img.get(cv2.CAP_PROP_FPS)
        # fps_placeholder.write(f"FPS:{fps}")
        # print(fps)
        labels_placeholder.table(names_table )
        print(names_table)
        
        if cv2.waitKey(1) & 0xFF == ord("q") or stop_button_pressed :  # press q to quit
            break
        
        # kill open cv things
    cap.release()
    cv2.destroyAllWindows()


def freeze_frame(frame):
    pass

# main()
stream()
