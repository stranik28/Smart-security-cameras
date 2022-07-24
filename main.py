from sort.sort import *

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from utils.augmentations import letterbox
from models.common import DetectMultiBackend
from utils.general import (check_img_size,  check_requirements, non_max_suppression,  scale_coords, xyxy2xywh)
from utils.torch_utils import select_device
import tracker
from agender import age_gender_detector

import cv2


cv2.imshow('1', cv2.imread('image.jpeg'))
cv2.waitKey(0)
cv2.destroyAllWindows()

mot_tracker = Sort()

weights = 'yolov5s.pt'
data = 'data/coco128.yaml'
imgsz = (480,288)
conf_thres = 0.25
iou_thres=0.45
max_det=300
device_img=False
device=''

tracker = tracker.EuclideanDistTracker()

# Попробовать поменять dnn на тру, это может помочь ускорить обработку
dnn=True
source = 'http://192.168.66.55:8080/video'
half=False
# фильтр по классу(если нужно вывести что-то конкретное)
classes = 0
augment=False  # augmented infereqnce
visualize=False
agnostic_nms=False


device = select_device(device)
model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
model.float()
model.eval()
stride, names, pt = model.stride, model.names, model.pt
imgsz = check_img_size(imgsz, s=stride)  # check image size

vid = cv2.VideoCapture('http://192.168.66.55:8080/video')
bs = 1
model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmu
dict_out = {}
cache_id = 0
while True:
    ret, image_show = vid.read()
    img = letterbox(image_show, imgsz, stride, auto=pt)[0]
    img = img.transpose((2, 0, 1))[::-1]
    img = np.ascontiguousarray(img)
    bs = 1  # batch_size
    model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup
    im = torch.from_numpy(img).to(device)
    im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
    im /= 255  # 0 - 255 to 0.0 - 1.0
    if len(im.shape) == 3:
        im = im[None]  # expand for batch dim
    pred = model(im, augment=augment, visualize=visualize)
    pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
    # detections = pred.cpu().numpy()
    for i, det in enumerate(pred):  # per image
        if len(det):
            det[:, :4] = scale_coords(im.shape[2:], det[:, :4], image_show.shape).round()
            cropList = []
            for *xyxy, _, _ in reversed(det):
                xyxy = torch.tensor(xyxy).view(-1, 4)
                b = xyxy2xywh(xyxy)
                # print(b)
                b = b.cpu().detach().numpy().tolist()
                b[0].append(0.8)
                cropList.append(b)
            for id,i in enumerate(cropList):
                # print()
                x,y,w,h,s = i[0]
                x1 = int(x-w/2)
                y1 = int(y-h/2)
                x2 = int(x+w/2)
                y2 = int(y+h/2)

                i[0] = [x1,y1,x2,y2,s]
                cropList[id] = i[0]
                # print(cropList[id])
            cropList = np.array(cropList)
            track_bbs_ids = mot_tracker.update(cropList)
            
            for j in range(len(track_bbs_ids.tolist())):
                coords = track_bbs_ids.tolist()[j]
                x1,y1,x2,y2 = int(coords[0]), int(coords[1]), int(coords[2]), int(coords[3])

                ori = image_show[y1:y2, x1:x2]
                
                name_idx = int(coords[4])
                print(name_idx)
                # if name_idx != cache_id and ori.any():
                #     print(age_gender_detector(ori))

                # print(name_idx)
                # color = colours[name_idx]
                # print(3)
                cv2.putText(image_show,str(name_idx),(x1,y1-5), cv2.FONT_HERSHEY_COMPLEX_SMALL,3, (255,255,255), 2)
                cv2.rectangle(image_show,(x1,y1),(x2,y2),(0,255,0),2)
        else:
            cropList = [0,0,0,0]
    cv2.imshow('1', cv2.resize(image_show,(640,480)))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break 
cv2.destroyAllWindows()
