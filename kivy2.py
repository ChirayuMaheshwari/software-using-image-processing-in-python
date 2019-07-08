import kivy
kivy.require("1.9.0")
import kivy
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.properties import ObjectProperty
from kivy.core.window import Window
from kivy.uix.popup import Popup
from Face12 import Q1
import numpy as np,cv2,os
from PIL import Image
from gtts import gTTS
import playsound
import pytesseract
# Used to display popup
VIDEO_TYPE = {
                    'avi': cv2.VideoWriter_fourcc(*'XVID'),
    #'mp4': cv2.VideoWriter_fourcc(*'H264'),
                    'mp4': cv2.VideoWriter_fourcc(*'XVID'),
        }

STD_DIMENSIONS =  {
            "480p": (640, 480),
            "720p": (1280, 720),
            "1080p": (1920, 1080),
            "4k": (3840, 2160),
        }
def get_video_type(filename):
        filename, ext = os.path.splitext(filename)
        if ext in VIDEO_TYPE:
            return  VIDEO_TYPE[ext]
        return VIDEO_TYPE['avi']
def get_dims(cap, res='1080p'):
    width, height = STD_DIMENSIONS["480p"]
    if res in STD_DIMENSIONS:
        width,height = STD_DIMENSIONS[res]
def get_video_type(filename):
        filename, ext = os.path.splitext(filename)
        if ext in VIDEO_TYPE:
            return  VIDEO_TYPE[ext]
        return VIDEO_TYPE['avi']
class CustomPopup1(Popup):
    pass
class SampBoxLayout(BoxLayout):
    # For checkbox
    checkbox_is_active = ObjectProperty(False)
    def checkbox_18_clicked(self, instance, value):
        if value is True:
            print("Checkbox Checked")
        else:
            print("Checkbox Unchecked")
 
    # For radio buttons
    blue = ObjectProperty(True)
    red = ObjectProperty(False)
    green = ObjectProperty(False)
 
    # For Switch
    def callback1(instance):
        #a=input()
        test_img=cv2.imread('akku.jpg')
        face_detected,gray_img=Q1.facedetection(test_img)
        faces,faceID=Q1.labels_for_training_data(r'C:\Users\chirayu maheshwari\images')
        face_recognizer=Q1.train_classifier(faces,faceID)
        name={1:"sachin",2:"virat kohli",3:"msd"}
        for face in face_detected:
            (x,y,w,h)=face
            roi_gray=gray_img[y:y+h,x:x+w]
            label,confidence=face_recognizer.predict(roi_gray)
            print(confidence)
            print(label)
            Q1.draw_rect(test_img,face)
            predicted_label=name[label]
            Q1.put_text(test_img,predicted_label,x,y)
        cv2.imshow("img",test_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    def callback5(instace):
        pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
        txt=pytesseract.image_to_string(Image.open("text_img.jpg"),lang="eng")
        img=cv2.imread('text_img.jpg')
        speech=gTTS(txt)
        print(txt)
        speech.save("1.mp3")
        playsound.playsound('1.mp3', True)
        cv2.imshow("img",img)
        cv2.waitKey(0)
		#cv2.destroyAllWindows()
    def callback3(instance):
        

# Set resolution for the video capture
# Function adapted from https://kirr.co/0l6qmh


# Standard Video Dimensions Sizes
        


# grab resolution dimensions and set video capture to it.


# Video Encoding, might require additional installs
# Types of Codes: http://www.fourcc.org/codecs.php
        


        cap = cv2.VideoCapture(0)
        fourcc=cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter('video.avi', fourcc, 20.0, (640,480))
        while True:
            ret, frame = cap.read()
            out.write(frame)
            cv2.imshow('frame',frame)
            if cv2.waitKey(20) & 0xFF == ord('q'):
                break


        cap.release()
        #out.release()
        cv2.destroyAllWindows()
    
    def callback4(instance):
        cap = cv2.VideoCapture('C:\\Users\\chirayu maheshwari\\video.avi')

#print cap.get(5) #to display frame rate of video
#print cap.get(cv2.cv.CV_CAP_PROP_FPS)

        while(cap.isOpened()): 
            ret, frame = cap.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #convert to grayscale
            cv2.imshow('frame',gray)
            cv2.waitKey(30)
            if cv2.waitKey(20) & 0xFF == ord('q'): 
                break

        cap.release()
        cv2.destroyAllWindows()
    def switch_on(self, instance, value):
        if value is True:
            print("Switch On")
        else:
            print("Switch Off")
 
    # Opens Popup when called
    def open_popup1(self):
        the_popup = CustomPopup1()
        the_popup.open()
    # For Spinner
    
 
 
class SampleApp(App):
    def build(self):
 
        # Set the background color for the window
        Window.clearcolor = (1, 1, 1, 1)
        return SampBoxLayout()
if __name__ == '__main__':
    sample_app = SampleApp()
    sample_app.run()
 
# ---------- sample.kv  ----------
 

 
