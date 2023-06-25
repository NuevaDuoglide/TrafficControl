import cv2
import numpy as np

#Web Camera
cap = cv2.VideoCapture('video.mp4')

#Inisiasi awal sebagai pembatas
lebar_minimal = 30
tinggi_minimal = 30

count_line = 550
# Inisiasi background
background = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40)

def center_handle(x,y,w,h):
    x1=int(w/2)
    y1=int(h/2)
    cx=x+x1
    cy=y+y1
    return cx,cy


detect = []
offset = 6 #Galat/Error
counter = 0

while True:
    ret, video = cap.read()
    gray = cv2.cvtColor(video, cv2.COLOR_BGR2GRAY)
    #Menambahkan gaussian blur untuk mengurangi noise
    blur = cv2.GaussianBlur(gray, (5,5), 5)
    
#   Menampilkan operasi morfologi untuk memperjelas tampilan gambar biner
    vid_sub = background.apply(blur)
    dilat = cv2.dilate(vid_sub, np.ones((5,5)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    dilated = cv2.morphologyEx(dilat, cv2.MORPH_CLOSE, kernel)
    contours, h = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    #Menghitung jumlah kendaraan dengan
    counter = len(contours)
    
#Loop untuk menggambar persegi di mobil yang telah dideteksi
    for (i, c) in enumerate(contours):
        (x,y,w,h) = cv2.boundingRect(c)
        val_counter = (w>=lebar_minimal) and (h>= tinggi_minimal)
        if not val_counter:
            continue
        cv2.rectangle(video,(x,y),(x+w,y+h),(0,255,0),2)


        for (x,y) in detect:
            if y<(count_line + offset) and  y>(count_line - offset):
                cv2.line(video, (25,count_line),(1200,count_line),(0,127,255), 3)
                detect.remove((x,y))

    cv2.imshow('Detector',video)

 #Teks total jumlah kendaraan
    cv2.putText(video, f"Jumlah Kendaraan: {counter}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

#Menampilkan jumlah kebdaraan
    cv2.imshow('Detector', video)

#Framerate dari video 
    if cv2.waitKey(30) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
