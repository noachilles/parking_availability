import argparse
import sys
from pathlib import Path
import os
import cv2
import time
import top_down
import pprint
import numpy as np

from models.common import DetectMultiBackend
from utils.general import (print_args, check_img_size)
from utils.torch_utils import select_device

import model_run as yolo
from is_occupied import parking

import torch
import torch.backends.cudnn as cudnn
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
points = np.load('point.npy')
park = parking(points, 2, 9)
park.divide_xy()

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf-thres', type=float, default=0.35, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=100, help='maximum detections per image')
    parser.add_argument('--view-img', default=True, action='store_true', help='show results')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_false', help='update all models')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    opt = parser.parse_args()
    # opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt

opt = parse_opt()
warp = top_down.Warping()

cap = cv2.VideoCapture('video/exp_2.mp4')
windows_name = 'result'
cv2.namedWindow(windows_name,flags=cv2.WINDOW_NORMAL)
cv2.resizeWindow(windows_name,width=960,height=540) 
cv2.moveWindow(windows_name,0,520)

# Load model
device = select_device('0')
imgsz=[640]
weights = 'best.pt'
model = DetectMultiBackend(weights=weights, device=device, dnn=True, fp16=True)
stride, names = model.stride, model.names
imgsz = check_img_size(imgsz, s=stride)  # check image size
imgsz *= 2 if len(imgsz) == 1 else 1  # expand

pretime = 0
count_frame = 0

if not cap.isOpened():
    raise IOError("Cannot open webcam")

while cap.isOpened():
    ret, frame = cap.read()
    curtime = time.time()
    key = cv2.waitKey(1)
    if ret:
        frame = cv2.resize(frame, (1920, 1080))
        frame = warp.trans_img(frame,width=1920,height=1080)    #탑-다운뷰 변환
        img, point = yolo.run(**vars(opt),source=frame,model = model,device=device,stride=stride,names=names,imgsz=imgsz)
        pre = park.spaces.copy()
        park.is_occupied(point)
        if count_frame % 30 == 0:
            send_dict = park.update_send(pre)
            if send_dict:
                pprint.pprint(send_dict)
        frame = park.draw_box(frame)
        sec = curtime - pretime
        pretime = curtime
        fps = 1/(sec)
        if count_frame % 5 == 0:
            str = "FPS : %0.1f" % fps
        if count_frame % 300 == 0:
            park.clear_count()
        #FPS 화면 표시
        cv2.putText(frame,str,(0,50),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),2)
        cv2.imshow(windows_name,frame)
        count_frame += 1
        
    
    if key & 0xFF == ord('s'):
        warp.save_H_mat(frame)

    
    #영상 종료
    if key & 0xFF == ord('e'):
        break

cap.release()
cv2.destroyAllWindows()
