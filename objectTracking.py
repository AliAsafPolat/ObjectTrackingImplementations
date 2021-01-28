# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 22:19:22 2020

@author: Ali_Asaf
"""

import numpy as np
import cv2
import argparse

#video = 'E:\\SPB_Data\\AraProje_VideoRenklendirme\\VideoRenklendirme\\demo\\example\\videolar\\tracking_Car.mp4'
video = 'E:\\SPB_Data\\AraProje_VideoRenklendirme\\VideoRenklendirme\\demo\\example\\videolar\\raspberry_tracking.mp4'

"""
# Instantiate OCV kalman filter
class KalmanFilter:

    kf = cv2.KalmanFilter(4, 2)
    kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
    kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)

    def Estimate(self, coordX, coordY):
        ''' This function estimates the position of the object'''
        measured = np.array([[np.float32(coordX)], [np.float32(coordY)]])
        self.kf.correct(measured)
        predicted = self.kf.predict()
        return predicted
"""

class KalmanFilter(object):
    def __init__(self, dt, u_x,u_y, std_acc, x_std_meas, y_std_meas):
        """
        :param dt: sampling time (time for 1 cycle)
        :param u_x: acceleration in x-direction
        :param u_y: acceleration in y-direction
        :param std_acc: process noise magnitude
        :param x_std_meas: standard deviation of the measurement in x-direction
        :param y_std_meas: standard deviation of the measurement in y-direction
        """
        # Define sampling time
        self.dt = dt
        # Define the  control input variables
        self.u = np.matrix([[u_x],[u_y]])
        # Intial State
        self.x = np.matrix([[0], [0], [0], [0]])
        # Define the State Transition Matrix A
        self.A = np.matrix([[1, 0, self.dt, 0],
                            [0, 1, 0, self.dt],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1]])
        # Define the Control Input Matrix B
        self.B = np.matrix([[(self.dt**2)/2, 0],
                            [0, (self.dt**2)/2],
                            [self.dt,0],
                            [0,self.dt]])
        # Define Measurement Mapping Matrix
        self.H = np.matrix([[1, 0, 0, 0],
                            [0, 1, 0, 0]])
        #Initial Process Noise Covariance
        self.Q = np.matrix([[(self.dt**4)/4, 0, (self.dt**3)/2, 0],
                            [0, (self.dt**4)/4, 0, (self.dt**3)/2],
                            [(self.dt**3)/2, 0, self.dt**2, 0],
                            [0, (self.dt**3)/2, 0, self.dt**2]]) * std_acc**2
        #Initial Measurement Noise Covariance
        self.R = np.matrix([[x_std_meas**2,0],
                           [0, y_std_meas**2]])
        #Initial Covariance Matrix
        self.P = np.eye(self.A.shape[1])
                
    def predict(self):
        # Refer to :Eq.(9) and Eq.(10)  in https://machinelearningspace.com/object-tracking-simple-implementation-of-kalman-filter-in-python/?preview_id=1364&preview_nonce=52f6f1262e&preview=true&_thumbnail_id=1795
        # Update time state
        #x_k =Ax_(k-1) + Bu_(k-1)     Eq.(9)
        self.x = np.dot(self.A, self.x) + np.dot(self.B, self.u)
        # Calculate error covariance
        # P= A*P*A' + Q               Eq.(10)
        self.P = np.dot(np.dot(self.A, self.P), self.A.T) + self.Q
        return self.x[0:2]

    def update(self, z):
        # Refer to :Eq.(11), Eq.(12) and Eq.(13)  in https://machinelearningspace.com/object-tracking-simple-implementation-of-kalman-filter-in-python/?preview_id=1364&preview_nonce=52f6f1262e&preview=true&_thumbnail_id=1795
        # S = H*P*H'+R
        S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R
        # Calculate the Kalman Gain
        # K = P * H'* inv(H*P*H'+R)
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))  #Eq.(11)
        self.x = np.round(self.x + np.dot(K, (z - np.dot(self.H, self.x))))   #Eq.(12)
        I = np.eye(self.H.shape[1])
        # Update error covariance matrix
        self.P = (I - (K * self.H)) * self.P   #Eq.(13)
        return self.x[0:2]


def kareCiz(event,x,y,flags,param):
    global refPt, cropping
    if event==cv2.EVENT_LBUTTONDOWN:
        refPt = [(x,y)]
        cropping=True
        print("left down : ",x, "|", y)
    elif event == cv2.EVENT_LBUTTONUP:
        refPt.append((x,y))
        cropping=False
        print("left up : " ,x, "|", y)
        

def cerceve_analizi_yap(frame, track_window, sender):
    x,y,w,h = track_window
    liste_x = []
    liste_y = []
    try:
        for i in range(y,min(int(y+h),int(frame.shape[0]-1))):
            for j in range(x,min(x+w,frame.shape[1]-1)):
                if(frame[i][j]>200): #threshold degerinden buyuk olanlari 255 yaptigi icin buyuk mu diye bakiyoruz.
                    liste_y.append(i)
                    liste_x.append(j)
                    
        sol_sinir = max(min(liste_x)-1,0)
        sag_sinir = min(max(liste_x)+1,frame.shape[1])
        ust_sinir = max(min(liste_y)-1,0)
        alt_sinir = min(max(liste_y)+1,frame.shape[0])
        
        yeni_track_window = sol_sinir,ust_sinir, sag_sinir - sol_sinir, alt_sinir - ust_sinir         
        return yeni_track_window        
    except:
        
        print("patladi : ",sender)
        return track_window

# Verilen videoda ve işaretli alana göre takip işlemini gerçekleştirir.
def video_oynat(cap,track_window,roi_hist):    
    # Window içerisinde hesaplama yaparken her defasında ana pencere kullanılacak. Sonrasında daraltmalar yapılacak.
    const_track_window = track_window
    x,y,const_w,const_h = const_track_window
    const_size = max(const_h, const_w)
    const_track_window = x,y,const_size,const_size
    # Setup the termination criteria, either 10 iteration or move by at least 1 pt
    term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )
    
    KF = KalmanFilter(0.1, 1, 1, 1, 0.1,0.1)
    #kfObj = KalmanFilter()
    predictedCoords = np.zeros((2, 1), np.float32)
    while(1):
        ret ,frame = cap.read()
    
        if ret == True:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            dst = cv2.calcBackProject([hsv],[0],roi_hist,[0,255],1)
    
            frame_meanshift = frame.copy()
            frame_kalman = frame.copy()
            
            # Lokasyonu almak için meanshift uygular.
            ret, track_window = cv2.meanShift(dst, track_window, term_crit)
            
            # Takip penceresinin konumu alınır.
            x,y,w,h = track_window
            
            (x_pred, y_pred) = KF.predict()
            (x1, y1) = KF.update((x,y))
            
            #print("Tahmin Edilen deger : ",x_pred,"|",y_pred)
            #print("Update Sonu deger : ", x1, "|", y1)
            
            # Tahmin edilen yeni koordinat degeri kalman filtresine verilir. 
            #predictedCoords = kfObj.Estimate(x, y)
            
            predictedCoords = x1,y1
            pp = predictedCoords[0].tolist()
            track_window_kalman = int(pp[0][0]),int(pp[0][1]),const_size,const_size
            
            # Her defasında ilgili koordinat değerlerinden bakıp daraltmayı ona göre yapsın.
            track_window = x,y,const_size,const_size
            
            # Arabanın şeklinin tam olarak belirlenmesi için bir dilate işlemi yapılır.
            kernel = np.ones((5,5), np.uint8)
            img_dilate = cv2.dilate(dst, kernel, iterations=1)
            kernel = np.ones((3,3), np.uint8)
            img_erode = cv2.erode(img_dilate, kernel, iterations=1)
            #kernel = np.ones((2,2), np.uint8)
            #img_dilate = cv2.dilate(img_erode, kernel, iterations=1)
            
            # Erosion ve dilation islemlerinden sonra daha net goruntu icin threshold uygulanir.
            ret,img_thresh = cv2.threshold(img_erode,64,255,cv2.THRESH_BINARY)
            
            
            
            # Verilen cerceve icerisindeki arac konumuna gore cerceve daraltilir.
            yeni_track = cerceve_analizi_yap(img_thresh, track_window,"meanshift")
            # Yeni konum ve cerceve genislik yukseklik bilgileri alinir.
            x,y,w,h = yeni_track
            
            yeni_track_kalman = cerceve_analizi_yap(img_thresh, track_window_kalman,"kalman")
            # Yeni konum ve cerceve genislik yukseklik bilgileri alinir.
            x_k,y_k,w_k,h_k = yeni_track_kalman
            
            img_meanShift = cv2.rectangle(frame_meanshift, (x,y), (x+const_w,y+const_h), 255,2)
            img_kalman = cv2.rectangle(frame_kalman, (x_k,y_k),(x_k + const_w, y_k + const_h), 255,1)
            cv2.imshow('thresholdImage', img_thresh)
            cv2.imshow('dilatedImage', img_dilate)
            cv2.imshow('erode image', img_erode)
            cv2.imshow('backpropScreen', dst)
            cv2.imshow('meanShift',img_meanShift)
            cv2.imshow('kalmanFilter', img_kalman)
    
            k = cv2.waitKey(60) & 0xff
            if k == 27:
                break
            
    
        else:
            break

refPt = []
cropping = False
cap = cv2.VideoCapture(video)
#cap = cv2.VideoCapture(0)


# take first frame of the video
ret,frame = cap.read()
print(frame.shape)
# setup initial location of window
# r,h,c,w - region of image
#           simply hardcoded the values

cv2.namedWindow("image")
cv2.setMouseCallback("image", kareCiz)
clone = frame.copy()

# display the image and wait for a keypress
cv2.imshow("image", frame)
key = cv2.waitKey(0)
	
	
#video_oynat(cap,track_window,roi_hist)
if len(refPt) == 2:
	#roi = clone[refPt[0][1]:refPt[1][1], refPt[0][0]:refPt[1][0]]
	#cv2.imshow("ROI", roi)
    # draw a rectangle around the region of interest
	cv2.rectangle(frame, refPt[0], refPt[1], (0, 255, 0), 2)
	cv2.imshow("image", frame)
	cv2.waitKey(0)
    
cv2.destroyAllWindows()
print(refPt[0][0],refPt[0][1],refPt[1][0]-refPt[0][0],refPt[1][1]-refPt[0][1] )
c,r,w,h = refPt[0][0],refPt[0][1],refPt[1][0]-refPt[0][0],refPt[1][1]-refPt[0][1]  
track_window = (c,r,w,h)

# set up the ROI fo"r tracking
roi = frame[r:r+h, c:c+w]
hsv_roi =  cv2.cvtColor(roi ,cv2.COLOR_BGR2HSV)
mask = cv2.inRange(hsv_roi, np.array((0., 60.,32.)), np.array((90.,126.,126.)))
# Meanshift calismasi icin cerceve ile belirlenmis alanin histogram bilgisi alinir.
roi_hist = cv2.calcHist([hsv_roi],[0],mask,[255],[0,255])
cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)
video_oynat(cap, track_window, roi_hist)

cv2.destroyAllWindows()
cap.release()