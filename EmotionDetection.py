# -*- coding: utf-8 -*-
"""
Created on Fri Nov 11 21:10:49 2022

@author: Santosh
"""

# -*- coding: utf-8 -*-
"""
"""
import cv2 as cv
import mediapipe as mp
import numpy as np


from sklearn.model_selection import train_test_split

from tensorflow.keras.models import load_model


import MeshInfo as mi
import ModelInfo as mo_i

COLOR_RED = (0,0,255)
COLOR_BLUE = (255,0,0)

COLOR_GREEN = (0,255,0)
COLOR_ORANGE = (39,125, 255)


#************ INITIALIZING PARAMETERS **********
#************ For Face mesh *****************
STATIC_IMAGE = False
MAX_NO_FACES = 2

DETECTION_CONFIDENCE = 0.6
TRACKING_CONFIDENCE = 0.5

COLOR_GREEN = (0,255,0)


#************ LOADING FACE MESH MODEL *******************
face_mesh = mp.solutions.face_mesh

face_model = face_mesh.FaceMesh(static_image_mode=STATIC_IMAGE,
                                max_num_faces= MAX_NO_FACES,
                                min_detection_confidence=DETECTION_CONFIDENCE,
                                min_tracking_confidence=TRACKING_CONFIDENCE)


#************ #################### *******************

no_of_classes = 3
target_size = 26

isTrain = False

#  TRAINING the model
if isTrain:

    dir_path = 'C:/Users/Santosh/OneDrive/Desktop/EMOTIONS'
    images, labels = mo_i.read_image_resize(dir_path);
    
    
    features =[]
    for image in images:
        image_src_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        
        outputs = face_model.process(image_src_rgb)
        
        angle_features = mi.get_angle_features(image, outputs, COLOR_GREEN, True)
        cv.imshow("IMAGE", image)
        if cv.waitKey(1000) & 255 == 27:
            break
        
        features.append(angle_features)
        outputs =[]

    x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size= 0.3, random_state = 1)

    x_train,y_train = mo_i.convert_to_tensor(x_train, y_train)
    x_test,y_test = mo_i.convert_to_tensor(x_test, y_test)
    
    
    model = mo_i.create_sequentia2(no_of_classes, x_train)
    
    model.summary()

    model.fit(x_train, y_train,
              validation_data = ( x_test, y_test),
              batch_size = 1, epochs = 2000 )
    
    model.save('OtherAccuracy.h5')
    
    #  PLOT and see how the model performs on this data
    mo_i.plot_model_summary(model)

else:
    
    # . LOADING the model
    model=load_model('90Accuracy.h5')


#***** VIDEO CAPTURE AND DISPLAY ******
capture = cv.VideoCapture(0)  
while True: 
    result, image = capture.read()
    if result:  
        try: 
                        
            image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB) 
            #-----------#############################----------------
            
            
            #-----------PROCESSING FACE---------------- 
            outputs = face_model.process(image_rgb)
            #-----------#############################----------------
            
            
            #-----------GET FACE ANGLE FEATURES---------------- 
            angle_features = mi.get_angle_features(image, outputs, COLOR_GREEN, False)
            #-----------#############################----------------
            
            #-----------RESHAPE AND CONVERT FEATURES TO TENSORS------
            feature_tensor_out = mo_i.reshape_array(angle_features)
            #-----------#############################----------------
            
            
            #-----------PREDICT THE CLASS--------------------------
            output = model.predict(feature_tensor_out)
            #-----------#############################----------------

            
            #-----------DECODE CLASS INDEX TO LABEL--------------------------
            # --- 0=ANGRY   1= HAPPY   2= SAD
            class_name, index = mo_i.decode_labels(output)
            
            points = mi.draw_landmarks(image, outputs, mi.FACE_RECT_REGIONS, COLOR_GREEN, False)
            
            
            #-------DRAWING THE FACE BOX-----------
            left = points[0][0]
            top = points[1][1]
            right = points[2][0]
            bottom = points[3][1]
            
            if index == 0:
                color_code = COLOR_RED
            if index == 1:
                color_code = COLOR_GREEN
            if index == 2:
                color_code = COLOR_ORANGE
            
            
            cv.rectangle(image, (left, top), (right, bottom), (255, 196, 55), 1)
            cv.line(image, (left, top), (left, top+10), color_code, 2)
            cv.line(image, (left, top), (left+10, top), color_code, 2)
            
            cv.line(image, (left, bottom), (left, bottom-10), color_code, 2)
            cv.line(image, (left, bottom), (left+10, bottom), color_code, 2)
            
            
            cv.line(image, (right, top), (right, top+10), color_code, 2)
            cv.line(image, (right, top), (right-10, top), color_code, 2)
            
            cv.line(image, (right, bottom), (right, bottom-10), color_code, 2)
            cv.line(image, (right, bottom), (right-10, bottom), color_code, 2)
            #-------##########################-----------
            
            
            #-------DRAWING TEXT OF THE CLASS OUTPUT-----------
            cv.putText(image, str(class_name), (left+5, top-10), cv.FONT_HERSHEY_PLAIN, 1, color_code, 2)
            #-------##########################-----------
            
            cv.imshow("FACE MESH", image)
            if cv.waitKey(30) & 255 == 27:
                break
        except:
         
            continue
        
#-------RELEASE CAMERA OBJECT AND CV OBJECT-----------        
capture.release()
cv.destroyAllWindows()


