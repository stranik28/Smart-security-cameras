
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from utils.augmentations import letterbox
from models.common import DetectMultiBackend
from utils.general import (check_img_size,  check_requirements, non_max_suppression,  scale_coords, xyxy2xywh)
from utils.torch_utils import select_device


@torch.no_grad()
def run1(
        weights='yolov5s.pt',  # model.pt path(s)
        source='data/images',  # file/dir/URL/glob, 0 for webcam
        data='data/coco128.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.40,  # confidence threshold
        iou_thres=0.55,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        im0s=False
):   
    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size
    img = letterbox(im0s, imgsz, stride, auto=pt)[0]
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
    for i, det in enumerate(pred):  # per image
        if len(det):
            det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0s.shape).round()
            cropList = []
            for *xyxy, _, _ in reversed(det):
                xyxy = torch.tensor(xyxy).view(-1, 4)
                b = xyxy2xywh(xyxy)
                b = b.cpu().detach().numpy().tolist()
                cropList.append(b)
            return cropList
            break 