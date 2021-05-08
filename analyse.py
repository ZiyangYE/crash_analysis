# -*- coding: utf-8 -*-
"""
Created on Sat May  8 03:13:23 2021

@author: JackC
"""

import numpy as np
import cv2
import os

from matplotlib import pyplot as plt


def video_to_pic(input_file,output_dir):
    cap = cv2.VideoCapture(input_file)
    cnt = 0
    if(not os.path.exists(output_dir)):
        os.makedirs(output_dir)
    
    ret, frame = cap.read()
    while ret:
        cv2.imwrite(output_dir+"/{0:04d}.jpg".format(cnt), frame)
        ret, frame = cap.read()
        cnt += 1

def pic_normalize(pic_file,target_file,out_file):
    pic1 = cv2.imread(target_file)
    pic2 = cv2.imread(pic_file)
    
    sift=cv2.xfeatures2d.SIFT_create()
    k1, d1 = sift.detectAndCompute(pic1, None)
    k2, d2 = sift.detectAndCompute(pic2, None)
    matcher = cv2.BFMatcher()
    mat = matcher.knnMatch(d1, d2, k=2)
    fmat = []
    for m1, m2 in mat:
        if m1.distance < 0.8 * m2.distance:
            fmat.append((m1.trainIdx, m1.queryIdx))
    km1 = np.float32([k1[i].pt for (t, i) in fmat])
    km2 = np.float32([k2[i].pt for (i, t) in fmat])
    H, t = cv2.findHomography(km2, km1, cv2.RANSAC,5.0)
    
    rst = cv2.warpPerspective(pic2, H, (pic2.shape[1], pic2.shape[0]))
    cv2.imwrite(out_file, rst)

def normalize(begin_num,end_num,input_dir,output_dir):
    cnt = 0
    if(not os.path.exists(output_dir)):
        os.makedirs(output_dir)
    for i in range(begin_num,end_num+1):
        pic_normalize(
            input_dir+"/{0:04d}.jpg".format(i),
            input_dir+"/{0:04d}.jpg".format(end_num),
            output_dir+"/{0:04d}.jpg".format(cnt)
        )
        cnt += 1
        
def labelize(begin_num,end_num,input_dir,output_dir):
    global cnt
    if(not os.path.exists(output_dir)):
        os.makedirs(output_dir)
    rst=[]
    cnt=0
    pic=cv2.imread(input_dir+"/{0:04d}.jpg".format(cnt))
    def mouse_down(event, x, y, flags, param):
        global cnt
        if event == cv2.EVENT_LBUTTONDOWN:
            rst.append((x,y))            
            pic=cv2.imread(input_dir+"/{0:04d}.jpg".format(cnt))
            cv2.circle(pic, rst[-1], 3, (0, 0, 255), thickness = -1)
            cv2.imwrite(output_dir+"/{0:04d}.jpg".format(cnt), pic)
            cnt+=1
            if(cnt==end_num+1):
                cv2.destroyAllWindows()
                return
            pic=cv2.imread(input_dir+"/{0:04d}.jpg".format(cnt))
            cv2.imshow("image", pic)
    
    cv2.namedWindow("image")
    cv2.setMouseCallback("image", mouse_down)
    cv2.imshow("image", pic)
    cv2.waitKey(0)
        
    return rst

def calcAndPlot(rst,ratio):
    xrst=[i[0] for i in rst]
    yrst=[i[1] for i in rst]
    
    xfrst=[]
    yfrst=[]
    
    for i in range(len(rst)-4):
        xfrst.append(sum(xrst[i:i+5])/5)
        yfrst.append(sum(yrst[i:i+5])/5)
    
    v=[]
    for i in range(len(xfrst)-1):
        v.append(((xfrst[i]-xfrst[i+1])**2+(yfrst[i]-yfrst[i+1])**2)**0.5)
    
    v=[i*ratio for i in v]
        
    x=[i/24 for i in range(len(v))]
    
    f1 = np.polyfit(x, v, 1)
    p1 = np.poly1d(f1)
    vfit=p1(x)
    
    plt.plot(x,v,label="raw")
    plt.plot(x,vfit,label="fit")
    plt.legend()
    plt.grid()
    plt.xlabel("time(second)")
    plt.ylabel("speed(kmph)")
    plt.ylim((0,200))
    plt.show()
    return v

if __name__ == '__main__':
    video_to_pic("base.mp4","pics/raw")
    normalize(94,117,"pics/raw","pics/normalized")
    rst=labelize(0,23,"pics/normalized","pics/labeled")
    calcAndPlot(rst,12.477)
