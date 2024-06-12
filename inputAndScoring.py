import schedule
import time
import tensorflow as tf 
import os
import numpy as np
from tensorflow.keras.models import Sequential
import cv2
#import datetime
import pandas as pd
#from gsheets import Sheets
#import csv
#import gspread
from google.oauth2.service_account import Credentials
import matplotlib.pyplot as plt
#import dataon

deviceid=0 # it depends on the order of USB connection. 
capture = cv2.VideoCapture(deviceid)

DIR = os.getcwd()

CATEGORIE = "image"
IMG_SIZE = 224
#カメラから取得した画像はimageディレクトリに保存される
path = os.path.join(DIR, CATEGORIE)
i=0
c=1
x=np.zeros((12,9))
 # 学習済みのモデル'saved_model/my_model1024.h5'を再現
new_model = tf.keras.models.load_model('saved_model/my_model1024.h5')

#sheet = dataon.on()


def label(ind):
    if ind == 0:
        return [100, 100]
    elif ind == 1:
        return [100, 50]
    elif ind == 2:
        return [100, 0]
    elif ind == 3:
        return [50, 100]
    elif ind == 4:
        return [50, 50]
    elif ind == 5:
        return [50, 0]
    elif ind == 6:
        return [0,100]
    elif ind == 7:
        return [0,50]
    else:
        return [0,0]

def scoring(acc):
    a = np.zeros((12,2))
    acc_max_ind = acc.argmax(axis = 1)
    for i in range(11):
        x = label(acc_max_ind[i])
        a[i][0] = x[0]
        a[i][1] = x[1]
    return_score = np.mean(a, axis=0)
    return return_score

def create_check_data():
    img_data = []
    for image_name in os.listdir(path):
        img_array = cv2.imread(os.path.join(path, image_name),)  # 画像読み込み
        img_resize_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))  # 画像のリサイズ
        img_data.append(img_resize_array)
    return img_data

def job():
    ret, frame = capture.read()
    if ret:
        global i
        global x
        global c
        #strdate=datetime.datetime.now().strftime('%Y%m%dT%H%M%S') 
        #fname="image_" + strdate + ".jpg"
        fname="image.jpg"
        cv2.imwrite('image/'+fname, frame) 
        print(fname + " is created.")
            
        #imgをモデル用にリサイズ
        img_data=create_check_data()
        #numpuに変換
        img_data = np.array(img_data)
        
        # 正答率を計算
        t = time.time()
        acc = new_model.predict(img_data)
        print("time taken by network : {:.3f}".format(time.time() - t))
        print(acc)
        print(i)
        x[i]=acc[0]
        i=(i+1)%12
        print(x)
        if i==0:
            print('x is full')   
            y=scoring(x)
            print(y)
            # sheet.update_cell(c,1,c)
            # sheet.update_cell(c, 2,round(y[1],0) )
            # sheet.update_cell(c, 3,round(y[0],0) )
            # #dataon.on(c,4,round(y[0],0))
            # #dataon.on(c,5,round(y[1],0))
            # #dataon.on(c,2,round(y[0],0)+10)
            # #dataon.on(c,3,round(y[1],0)-5)
            # #dataon.on(c,6,round(y[0],0)-5)
            # #dataon.on(c,7,round(y[1],0)+5)
            c=c+1
#do job every 10 seconds

schedule.every(1/30).minutes.do(job)
while True:
    schedule.run_pending()
    time.sleep(1)

