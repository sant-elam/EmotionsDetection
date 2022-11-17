# -*- coding: utf-8 -*-
"""
Created on Sun Nov 13 22:07:31 2022

@author: Santosh
"""

import math
import cv2 as cv

LIPS=[ 61, 146, 91, 181, 84, 17, 314, 405, 321, 375,291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95,
       185, 40, 39, 37,0 ,267 ,269 ,270 ,409, 415, 310, 311, 312, 13, 82, 81, 42, 183, 78 ]


RIGHT_EYEBROW=[ 70, 63, 105, 66, 107, 55, 65, 52, 53, 46 ]
LEFT_EYEBROW =[ 336, 296, 334, 293, 300, 276, 283, 282, 295, 285 ]

LEFT_EYE_MESH = [263, 249, 390, 373, 374, 380, 381, 382, 362,263, 466, 388, 387, 386, 385, 384, 398, 362]
RIGHT_EYE_MESH = [33, 7, 163 ,144, 145, 153, 154, 155, 133,33, 246, 161, 160, 159, 158, 157, 173, 133]

FACE=[ 10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400,
       377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103,67, 109]


LIPS_ANGLE =[61, 0, 291, 17]

RIGHT_EYEBROW_ANGLE=[ 46, 52, 55 ]
LEFT_EYEBROW_ANGLE =[ 276, 282, 285 ]

LEFT_EYE_ANGLE = [263, 386, 362, 374]
RIGHT_EYE_ANGLE = [33, 159, 145, 133]

LEFT_CHEEK = [50, 206]
RIGHT_CHEEK = [280, 426]


NOSE_ANGLE = [129, 4, 358]

EYEBROW_CONNECT = [122, 351]



A_POINTS = [0,61,17]
B_POINTS = [61,17,291]
C_POINTS = [61, 0, 291]
D_POINTS = [291, 426, 358]
E_POINTS = [17, 291, 4]
F_POINTS = [61, 4, 291]
G_POINTS = [61, 50, 145]
H_POINTS = [358, 280, 374]
I_POINTS = [50, 129, 4]
J_POINTS = [122, 4, 351]
K_POINTS = [33, 46, 52]
L_POINTS = [386, 263, 374]
M_POINTS = [285, 282, 276]
N_POINTS = [46, 52, 55]
O_POINTS = [52, 55, 122]
P_POINTS = [282, 285, 351]
Q_POINTS = [159, 33, 145]
R_POINTS = [159, 133, 145]

S_POINTS = [33, 159, 133]
T_POINTS = [33, 145, 133]
U_POINTS = [386, 362, 374]
V_POINTS = [386, 263, 374]

W_POINTS = [362, 386, 263]
X_POINTS = [362, 374, 263]

Y_POINTS = [33, 145, 122]
Z_POINTS = [351, 374, 263]

ALL_ANGLE_POINTS = [A_POINTS, B_POINTS, C_POINTS, D_POINTS, E_POINTS, F_POINTS,
                    G_POINTS, H_POINTS, I_POINTS, J_POINTS, K_POINTS, L_POINTS,
                    M_POINTS, N_POINTS, O_POINTS, P_POINTS, Q_POINTS, R_POINTS,
                    S_POINTS, T_POINTS, U_POINTS, V_POINTS, W_POINTS, X_POINTS,
                    Y_POINTS, Z_POINTS]

FACE_RECT_REGIONS = [234, 10, 454, 152]



def draw_landmarks(image, outputs, land_mark, color, draw=True):
    height, width =image.shape[:2]
            
    points =[]
    for face in land_mark:
        point = outputs.multi_face_landmarks[0].landmark[face]
        
        point_scale = ((int)(point.x * width), (int)(point.y*height))
        
        print(face)
        if draw:
            cv.circle(image, point_scale, 2, color, 1)
            cv.putText(img=image, text=str(face), org=point_scale, fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=0.35, color=(0, 255, 0),thickness=1)

        
        points +=[point_scale]
        
    return points
        
def draw_all_face_points(image, outputs, color, draw=True):
    
    height, width =image.shape[:2]
    
    points =[]

    for land_mark_point in outputs.multi_face_landmarks[0].landmark:
        x,y,_ = land_mark_point.x, land_mark_point.y, land_mark_point.z
    
        point_scale = ((int)(x * width), (int)(y*height))
        if draw:
            cv.circle(image, point_scale, 1, color, 1)
        
        points +=[point_scale]
        
        
    return points
             
def draw_face_angle_points(image, outputs,land_mark, color, draw=True):
    
    height, width =image.shape[:2]
    
    points =[]

    for face in land_mark:
        point = outputs.multi_face_landmarks[0].landmark[face]
        
        point_scale = ((int)(point.x * width), (int)(point.y*height))
        
        
        points +=[point_scale]
        
        if draw:
            cv.circle(image, point_scale, 3, color, 1)
        
        
    #print(points)
    
    if draw:
        cv.circle(image, points[1], 5, color, 1)
        
    distanceA = math.sqrt((points[1][0] - points[0][0])**2 + (points[1][1] - points[0][1])**2)
    
    distanceB = math.sqrt((points[2][0] - points[1][0])**2 + (points[2][1] - points[1][1])**2)  
    
    distanceC = math.sqrt((points[2][0] - points[0][0])**2 + (points[2][1] - points[0][1])**2) 
        
    angleC = math.acos((distanceA**2 + distanceB**2 -distanceC**2) / (2 * distanceA * distanceB))
    
    
    result = angleC * (180.0 / math.pi)
    #print(result)
    return angleC

def get_angle_features(image, outputs, color, draw=True):
    angle_features =[]
    if outputs.multi_face_landmarks:   
        #draw_landmarks(image_src, outputs, FACE, COLOR_GREEN)
    
        for angle_points in ALL_ANGLE_POINTS:
             #draw_landmarks(image, outputs, FACE, COLOR_RED)
             angle_feature = draw_face_angle_points(image, outputs, angle_points, color, draw)
             angle_features.append(angle_feature)
             
    return angle_features