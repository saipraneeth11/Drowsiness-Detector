

import numpy as np
import dlib 
import cv2

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
#here the detector detects the number of faces in the image and returns the bounding box
#here we use the predictor to get the 68 co-ordinates 
 

  
def rect_to_bb(face):
    x = face.left()
    y = face.top()
    w = face.right()-x
    h = face.bottom()-y
    return (x,y,w,h)    


def to_nparr(shape,dtype = 'int'):
    cord = np.zeros((68,2),dtype = dtype)
    #this is a 68*2 array now we have to add the values into it 
    for i in range(0,68):
        cord[i] = (shape.part(i).x , shape.part(i).y)
        
    return cord 




#main programm 
cam = cv2.VideoCapture(0)
counter = 0
while(cam.isOpened()):
    ret , frame = cam.read()
    #now flip this image 
    frame = cv2.flip(frame,1)
    grayimg = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces = detector(grayimg,1)
    
    for i in range(0,len(faces)):   
        (x,y,w,h) = rect_to_bb(faces[i])
        shape = predictor(grayimg,faces[i])
        #convert to array 
        shape = to_nparr(shape)
        
        leye = shape[36:42]
        reye = shape[42:48]
        
        
        for (x,y) in leye:
            
            cv2.circle(frame , (x,y), 1 , (0,0,255),-1)
        for (x,y) in reye:
            cv2.circle(frame , (x,y), 1 , (0,255,0),-1)
        r1 = np.linalg.norm(shape[43]-shape[47])
        r2 = np.linalg.norm(shape[44]-shape[46])
        l1 = np.linalg.norm(shape[37]-shape[41])
        l2 = np.linalg.norm(shape[38]-shape[40])    
        r3 = np.linalg.norm(shape[42]-shape[45])
        l3 = np.linalg.norm(shape[36]-shape[39])
        
        lval = (l1 + l2 )/(2*l3)
        rval = (r1 + r2 )/(2*r3)
        
        cv2.putText(frame,'lval {}'.format(lval),(300,30),cv2.FONT_HERSHEY_DUPLEX,0.7,(0,0,255),2)
        cv2.putText(frame,'rval {}'.format(rval),(300,60),cv2.FONT_HERSHEY_DUPLEX,0.7,(0,0,255),2)    
        
        if(lval<0.24 and rval <0.24):
            counter = counter + 1
        if(counter > 20):
            print('Drowsy')
            counter = 0
        
        
        
        
    cv2.imshow('drowsiness',frame)    
    #this is the delay element of 10ms
    key = cv2.waitKey(20)
    if key == 27 :
        break
    
    
    
cam.release()
cv2.destroyAllWindows()
    

    
