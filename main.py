import pickle
import cv2
import mediapipe as mp
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from tensorflow import keras

CamH_id = 0 # Camera ID for hand gesture
CamL_id = 1 # Camera ID for left camera
CamR_id = 2 # Camera ID for right camera

CamH = cv2.VideoCapture(CamH_id)
CamL = cv2.VideoCapture(CamL_id)
CamR = cv2.VideoCapture(CamR_id)

# Load the model for hand gesture recognition
model_dict = pickle.load(open('./hand_gesture_model.p', 'rb'))
model = model_dict['model']
labels_dict = {0: 'forward', 1: 'stop', 2: 'turn left', 3:'turn right'}

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Reading the mapping values for stereo image rectification
cv_file = cv2.FileStorage("F:/Assignments 7 sem/FYP/smart_wheel_chair/data_ob/stereo_rectify_maps.xml", cv2.FILE_STORAGE_READ)
Left_Stereo_Map_x = cv_file.getNode("Left_Stereo_Map_x").mat()
Left_Stereo_Map_y = cv_file.getNode("Left_Stereo_Map_y").mat()
Right_Stereo_Map_x = cv_file.getNode("Right_Stereo_Map_x").mat()
Right_Stereo_Map_y = cv_file.getNode("Right_Stereo_Map_y").mat()
cv_file.release()

disparity = None
depth_map = None

# These parameters can vary according to the setup
max_depth = 400 # maximum distance the setup can measure (in cm)
min_depth = 50 # minimum distance the setup can measure (in cm)
depth_thresh = 200.0 # Threshold for SAFE distance (in cm)

# Reading the stored the StereoBM parameters
cv_file = cv2.FileStorage("F:/Assignments 7 sem/FYP/smart_wheel_chair/data/depth_estmation_params_py.xml", cv2.FILE_STORAGE_READ)
numDisparities = int(cv_file.getNode("numDisparities").real())
blockSize = int(cv_file.getNode("blockSize").real())
preFilterType = int(cv_file.getNode("preFilterType").real())
preFilterSize = int(cv_file.getNode("preFilterSize").real())
preFilterCap = int(cv_file.getNode("preFilterCap").real())
textureThreshold = int(cv_file.getNode("textureThreshold").real())
uniquenessRatio = int(cv_file.getNode("uniquenessRatio").real())
speckleRange = int(cv_file.getNode("speckleRange").real())
speckleWindowSize = int(cv_file.getNode("speckleWindowSize").real())
disp12MaxDiff = int(cv_file.getNode("disp12MaxDiff").real())
minDisparity = int(cv_file.getNode("minDisparity").real())
M = cv_file.getNode("M").real()
cv_file.release()

output_canvas = None

# Creating an object of StereoBM algorithm
stereo = cv2.StereoBM_create()

def obstacle_avoid(st):

    # Mask to segment regions with depth less than threshold
    mask = cv2.inRange(depth_map,10,depth_thresh)

    # Check if a significantly large obstacle is present and filter out smaller noisy regions
    if np.sum(mask)/255.0 > 0.01*mask.shape[0]*mask.shape[1]:

        # Contour detection 
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cnts = sorted(contours, key=cv2.contourArea, reverse=True)
        
        # Check if detected contour is significantly large (to avoid multiple tiny regions)
        if cv2.contourArea(cnts[0]) > 0.01*mask.shape[0]*mask.shape[1]:

            x,y,w,h = cv2.boundingRect(cnts[0])

            # finding average depth of region represented by the largest contour 
            mask2 = np.zeros_like(mask)
            cv2.drawContours(mask2, cnts, 0, (255), -1)

            # Calculating the average depth of the object closer than the safe distance
            depth_mean, _ = cv2.meanStdDev(depth_map, mask=mask2)
            
            # Display warning text
            cv2.putText(output_canvas, "WARNING !", (x+5,y-40), 1, 2, (0,0,255), 2, 2)
            cv2.putText(output_canvas, "Object at", (x+5,y), 1, 2, (100,10,25), 2, 2)
            cv2.putText(output_canvas, "%.2f cm"%depth_mean, (x+5,y+40), 1, 2, (100,10,25), 2, 2)
            state = 1

    else:
        cv2.putText(output_canvas, "SAFE!", (100,100),1,3,(0,255,0),2,3)
        state = st

    cv2.imshow('output_canvas',output_canvas)
    return state
    
while True:
    
    data_aux = []
    x_ = []
    y_ = []
    state = 1
    
    ret, imgH = CamH.read()
    retR, imgR = CamR.read()
    retL, imgL = CamL.read()
    
    # Shape of hand gesture image, into RGB
    H, W, _ = imgH.shape
    frame_rgb = cv2.cvtColor(imgH, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                imgH,  # image to draw
                hand_landmarks,  # model output
                mp_hands.HAND_CONNECTIONS,  # hand connections
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

        for hand_landmarks in results.multi_hand_landmarks:
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y

                x_.append(x)
                y_.append(y)

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))

        x1 = int(min(x_) * W) - 10
        y1 = int(min(y_) * H) - 10

        x2 = int(max(x_) * W) - 10
        y2 = int(max(y_) * H) - 10

        try:
            X=np.asarray([data_aux])     
            prediction = model.predict(X)
            onehot_encoder = OneHotEncoder(sparse_output=False)
            onehot_encoded = onehot_encoder.fit_transform(np.array([0,1,2,3]).reshape(-1, 1))
            prediction = onehot_encoder.inverse_transform(prediction)
            state = int(prediction[0])
            print(int(prediction[0]))
        
        except:
            pass
        
    
    elif retL and retR:
        
        output_canvas = imgL.copy()

        imgR_gray = cv2.cvtColor(imgR,cv2.COLOR_BGR2GRAY)
        imgL_gray = cv2.cvtColor(imgL,cv2.COLOR_BGR2GRAY)

        # Applying stereo image rectification on the left image
        Left_nice = cv2.remap(imgL_gray,
                            Left_Stereo_Map_x,
                            Left_Stereo_Map_y,
                            cv2.INTER_LANCZOS4,
                            cv2.BORDER_CONSTANT,
                            0)
        
        # Applying stereo image rectification on the right image
        Right_nice = cv2.remap(imgR_gray,
                            Right_Stereo_Map_x,
                            Right_Stereo_Map_y,
                            cv2.INTER_LANCZOS4,
                            cv2.BORDER_CONSTANT,
                            0)

        # Setting the updated parameters before computing disparity map
        stereo.setNumDisparities(numDisparities)
        stereo.setBlockSize(blockSize)
        stereo.setPreFilterType(preFilterType)
        stereo.setPreFilterSize(preFilterSize)
        stereo.setPreFilterCap(preFilterCap)
        stereo.setTextureThreshold(textureThreshold)
        stereo.setUniquenessRatio(uniquenessRatio)
        stereo.setSpeckleRange(speckleRange)
        stereo.setSpeckleWindowSize(speckleWindowSize)
        stereo.setDisp12MaxDiff(disp12MaxDiff)
        stereo.setMinDisparity(minDisparity)

        # Calculating disparity using the StereoBM algorithm
        disparity = stereo.compute(Left_nice,Right_nice)
        # NOTE: compute returns a 16bit signed single channel image,
        # CV_16S containing a disparity map scaled by 16. Hence it 
        # is essential to convert it to CV_16S and scale it down 16 times.

        # Converting to float32 
        disparity = disparity.astype(np.float32)

        # Normalizing the disparity map
        disparity = (disparity/16.0 - minDisparity)/numDisparities
        
        depth_map = M/(disparity) # for depth in (cm)

        mask_temp = cv2.inRange(depth_map,min_depth,max_depth)
        depth_map = cv2.bitwise_and(depth_map,depth_map,mask=mask_temp)

        state = obstacle_avoid(state)
        
        cv2.resizeWindow("disp",700,700)
        cv2.imshow("disp",disparity)

        if cv2.waitKey(1) == 27:
            break
    
    else:
        CamL = cv2.VideoCapture(CamL_id)
        CamR = cv2.VideoCapture(CamR_id)

    cv2.imshow('frame', imgH)
    cv2.waitKey(1)