import streamlit as st
import cv2
import numpy as np 
from ultralytics import YOLO
from PIL import Image
import os

#Setting up the title of the Streamlit app

st.title("YOLO Application(Number Plate Detection)")

#Allow access to upload a video or image file
uploaded_file = st.file_uploader("Upload Image or Video Files" , type = ['jpg' , 'png' , 'jpeg' , 'mp4','mkv'] )

#Loading the existing model

try:
    model = YOLO("./Number_Plate_Detection_model(Yolov8).pt")
except Exception as e:
    print(f"Unsupported or corrupted Model : {e}")


def predict_save_image(test_path , output_image_path):

    """
    This function predicts and saves the bounding boxes region on the given test image file using the trained YOLOv8 model.
    Parameters:
    test_path: Path to the test image file.
    output_image_path: Path, in which output image file will be saved.

    Returns:The path to the saved output image file.
    """

    try:
        results = model.predict(test_path , device = 'cpu')
        img = cv2.imread(test_path)
        img = cv2.cvtColor(img , cv2.COLOR_BGR2RGB)
        
        for result in results:
            for box in result.boxes:
                
                x1,y1,x2,y2 = map(int , box.xyxy[0])
                confidence = box.conf[0]
                cv2.rectangle(img , (x1,y1) , (x2,y2) , (0,255,0) , 2)
                cv2.putText(img , f'{confidence*100 :.2f}%' , (x1,y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9 , (255,0,0) , 2)  
                    
        img = cv2.cvtColor(img , cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_image_path , img)
        return output_image_path

    except Exception as e:
        print(f"Error in processing input image file {e}")

def predict_video(test_video , output_video_path):

    """
    This function predicts and saves the bounding boxes on the given test video using the trained YOLO model.
    Parameters:
    test_video: Path to the test video file.
    output_video_path:Path, in which output video file will be saved.
    Returns:
    The path to the saved output video file.
    """

    try:
        captured_video = cv2.VideoCapture(test_video)

        if not captured_video.isOpened():
            st.error(f"Error in processing video : {test_video}")
            return None
        
        frame_width = int(captured_video.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(captured_video.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fps = int(captured_video.get(cv2.CAP_PROP_FPS))

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path , fourcc , fps , (frame_width , frame_height))

        while captured_video.isOpened():
            ret , frame = captured_video.read()
            if not ret:
                break 

            rgb_frame = cv2.cvtColor(frame , cv2.COLOR_BGR2RGB)
            results = model.predict(rgb_frame , device = 'cpu')

            for result in results:

                for box in result.boxes:
                
                    x1,y1,x2,y2 = map(int , box.xyxy[0])
                    confidence = box.conf[0]
                    cv2.rectangle(frame , (x1,y1) , (x2,y2) , (0,255,0) , 2)
                    cv2.putText(frame , f'{confidence*100 :.2f}%' , (x1,y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9 , (255,0,0) , 2)  

            out.write(frame)
        captured_video.release()
        out.release()
        return output_video_path
    
    except Exception as e:
         print(f"Error in processing input video file {e}")



def process_media(input_path , output_path):

    """
    This function processes the uploaded media file (image or video) and returns the path to the saved output file.
    Parameters:
    input_path: Path to the input media file.
    output_path: Path to save the output media file.

    Returns:
    The path to the saved output media file.
    """

    file_ext = os.path.splitext(input_path)[1].lower()
    if file_ext in ['.mkv' , '.mp4' , '.mov']:
        return predict_video(input_path , output_path)
    elif file_ext in ['.jpg' , '.jpeg' , '.png']:
        return predict_save_image(input_path , output_path)
    else:
        st.error(f"Unsupported format : {file_ext}")
        return None


if uploaded_file is not None:
    input_path = os.path.join("temp" , uploaded_file.name)
    output_path = os.path.join("output" , uploaded_file.name)

    with open(input_path , 'wb') as f:
        f.write(uploaded_file.getbuffer())
    st.write("Processing the Input file for the Output")
    result_path = process_media(input_path , output_path)

    ##logic for prediction
    if result_path :
        if input_path.endswith(('.mkv' , '.mp4' , '.mov')):

            video_file = open(result_path , 'rb')
            video_bytes = video_file.read()
            st.video(video_bytes , format="video/mp4")
        else:
            st.image(result_path)
