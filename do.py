# -*- coding: utf-8 -*-
import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('true1.png')

imgray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

sizeAll = imgray.shape
areaAll = sizeAll[0]*sizeAll[1]
# areaall2 = cv2.contourArea(imgray)
ret,thresh = cv2.threshold(imgray,127,255,0)
contours,hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)


#TODO
#  考虑1轮廓识别看能不能更多，先获取更多的轮廓 再来落地
#  考虑2轮廓筛选，除了第一层外获取获取第二层的轮廓？可以测试下
#  考虑3轮廓筛选，轮廓的宽高比的筛选。
#  考虑4轮廓形状筛选：轮廓的4个方位的顶点一定是构成一个矩形
#  考虑5轮廓面积？？？如何确定阈值或者动态获取阈值？  参考获取同一个图中的相似同层级标志块的面积做为过滤面积？
#  考虑6像素筛选的优化？
newcontours =[]
areaList = []

isUseSmall = False
isUseMean = False
#基于面积筛选后再画
i = 0
for c in contours:
    area = cv2.contourArea(c)
    #表示面积足够大
    if(area >= areaAll * 0.00):
        mask = np.zeros(imgray.shape, np.uint8)
        # 这里一定要使用参数-1，绘制填充的轮廓
        cv2.drawContours(mask, [c], 0, 255, -1)
        pixelpoints = np.transpose(np.nonzero(mask))
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(imgray, mask=mask)
        print(min_val, max_val, min_loc, max_loc)
        mean_val = cv2.mean(imgray, mask=mask)
        print(mean_val)
        print(area)
        # print(pixelpoints)
        # cv2.imshow('mask', mask)
        # cv2.waitKey()
        print("####")
        print(i)
        print(len(c))
        print(len(contours[i]))
        print(hierarchy[0,i])
        print("####")
        #表示最小轮廓
        if(hierarchy[0,i,2]==-1 or not isUseSmall):
            #表示均值靠近全黑或者全白
            if(mean_val[0]>=240 or mean_val[0]<=15 or not isUseMean):
                newcontours.append(c)
                areaList.append(area)
    i = i+1

#绘制独立轮廓，如第四个轮廓
# imag = cv2.drawContours(img,contours,-1,(0,255,0),3)
imagnew = cv2.drawContours(img,newcontours,-1,(0,255,0),3)
#但是大多数时候，下面方法更有用
# imag = cv2.drawContours(img,contours,3,(0,255,0),3)

print(len(contours))
print(len(newcontours))

print(areaAll)
print(areaList)

while(1):
    cv2.imshow('img',img)
    cv2.imshow('imgray',imgray)
    # cv2.imshow('image',image)
    # cv2.imshow('imag',imag)
    cv2.imshow('imagnew',imagnew)
    if cv2.waitKey(1) == ord('q'):
        break
cv2.destroyAllWindows()