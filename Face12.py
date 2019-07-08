import numpy as np
import cv2
import os
class Q1:
 def facedetection(test_img):
    face_cascade=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    gray_img=cv2.cvtColor(test_img,cv2.COLOR_BGR2GRAY)
    faces=face_cascade.detectMultiScale(gray_img,1.3,5)
    return faces,gray_img
 def labels_for_training_data(directory):
   faces=[]
   faceID=[]
   for path,subdirnames,filenames in os.walk(directory):#gives path subdirectory and filenames
      for filename in filenames:
          #if filename.startswitch("."):
            #print("skipping system file")
            #continue
          id=os.path.basename(path)
          img_path=os.path.join(path,filename)
          #print("img_path:",img_path)
          #print("id:",id)
          test_img=cv2.imread(img_path)
          if test_img is None:
            print("Image not loaded")
            continue
          faces_rect,gray_img=Q1.facedetection(test_img)
          if len(faces_rect)!=1:
            continue#only single person image is transfer
          (x,y,w,h)=faces_rect[0]
          roi_gray=gray_img[y:y+w,x:x+h]
          faces.append(roi_gray)
          faceID.append(int(id))
   return faces,faceID

 def train_classifier(faces,faceID):
  face_recognizer=cv2.face.LBPHFaceRecognizer_create()
  face_recognizer.train(faces,np.array(faceID))
  return face_recognizer
 def draw_rect(test_img,face):
  (x,y,w,h)=face
  cv2.rectangle(test_img,(x,y),(x+w,y+h),(255,0,0),1)
 def put_text(test_img,text,x,y):
   cv2.putText(test_img,text,(x,y),cv2.FONT_HERSHEY_DUPLEX,1,(255,0,0),1)       
 
