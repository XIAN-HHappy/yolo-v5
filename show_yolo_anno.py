#-*-coding:utf-8-*-
# date:2019-08
# Author: Eric.Lee
# function: show yolo datasets anno

import cv2
import os
import numpy as np

if __name__ == "__main__":

    path='../datasets_fusion_hand_train/anno/train.txt' # 数据集文件路径

    label_map = ["hand"]

    with open(path, 'r') as file:
        img_files = file.read().splitlines()
        img_files = list(filter(lambda x: len(x) > 0, img_files))

    label_files = [
        x.replace('images', 'labels').replace("JPEGImages", 'labels').replace('.bmp', '.txt').replace('.jpg', '.txt').replace('.png', '.txt')
        for x in img_files]

    for i in range(len(img_files)):
        print(img_files[i])
        img = cv2.imread(img_files[i])
        w = img.shape[1]
        h = img.shape[0]

        label_path = label_files[i]
        print(i,label_path)
        if os.path.isfile(label_path):
            with open(label_path, 'r') as file:
                lines = file.read().splitlines()

            x = np.array([x.split() for x in lines], dtype=np.float32)
        for k in range(len(x)):
            anno = x[k]
            label = int(anno[0])
            x1 = int((float(anno[1])-float(anno[3])/2)*w)
            y1 = int((float(anno[2])-float(anno[4])/2)*h)

            x2 = int((float(anno[1])+float(anno[3])/2)*w)
            y2 = int((float(anno[2])+float(anno[4])/2)*h)

            cv2.rectangle(img, (x1,y1), (x2,y2), (255,30,30), 2)

            cv2.putText(img, ("%s" %(str(label_map[label]))), (x1,y1),\
            cv2.FONT_HERSHEY_PLAIN, 2.5, (0, 255, 55), 6)
            cv2.putText(img, ("%s" %(str(label_map[label]))), (x1,y1),\
            cv2.FONT_HERSHEY_PLAIN, 2.5, (0, 55, 255), 2)
            # cv2.circle(img, (x1,y1), 4, (0,255,225), 6)

        cv2.namedWindow('image',0)
        cv2.imshow('image',img)
        if cv2.waitKey(1) == 27:
            break
    cv2.destroyAllWindows()
