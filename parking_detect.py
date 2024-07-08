import argparse
from itertools import count
import sys
from pathlib import Path
import os, time,  cv2
from pprint import pprint 

import top_down
import model_run as yolo
from ing_occupied import parking

from models.common import DetectMultiBackend
from utils.general import (print_args, check_img_size)
from utils.torch_utils import select_device

import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
from threading import Thread, Lock

################### firebase connection ######################
class Worker(Thread):
    def __init__(self, lst_lots):
        Thread.__init__(self)

        self.lst_lots = lst_lots
        self.lst_to_be_updated = []
        self.lock = Lock()

        self.init_db()

    def init_db(self):
        cred = credentials.Certificate('00.json')
        firebase_admin.initialize_app(cred, {'projectId': '',})
        db = firestore.client()

        self.doc_refs = {}
        self.docs = {}
        for item in self.lst_lots:
            self.doc_refs[item] = db.collection(u'map').document(u'kmou').collection(u'lot').document(item)
            self.docs[item] = self.doc_refs[item].get().to_dict()
        
    def run(self):
        item = None
        while True:
            with self.lock:
                if self.lst_to_be_updated:
                    item = self.lst_to_be_updated[0]
                    self.lst_to_be_updated.remove(item)
        
            if item is not None:
                print(f'set-> {item}')
                self.docs[item[0]]['occupied'] = item[1]
                self.doc_refs[item[0]].set(self.docs[item[0]])
                item = None
            
            time.sleep(0.1)

    def update(self, lot):
        with self.lock:
            self.lst_to_be_updated.append(lot)
##############################################


FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

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

# Load model
device = select_device('0')
imgsz=[640]
weights = 'best.pt'
model = DetectMultiBackend(weights=weights, device=device, dnn=True, fp16=True)
stride, names = model.stride, model.names
imgsz = check_img_size(imgsz, s=stride)  # check image size
imgsz *= 2 if len(imgsz) == 1 else 1  # expand

# webcam에서 영상 불러오기
cap =cv2.VideoCapture(0)
print(cap.get(cv2.CAP_PROP_FRAME_COUNT))
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
print('original size: %d, %d' % (width, height))

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
print('changed size: %d, %d' % (width, height))

windows_name = 'result'
origin_name = 'origin'
cv2.namedWindow(windows_name,flags=cv2.WINDOW_NORMAL)
cv2.resizeWindow(windows_name,width=800,height=540)
cv2.moveWindow(windows_name,0,0)

cv2.namedWindow(origin_name,flags=cv2.WINDOW_NORMAL)
cv2.resizeWindow(origin_name,width=980,height=540)
cv2.moveWindow(origin_name,0,720)

#주차 칸수 가로, 세로
n, m = 9,2

#image size
w = 1920    
h = 1080

warp = top_down.Warping()
points = warp.return_pts2()
park = parking(points, n, m)
park.divide_xy()

pretime = 0
count_frame = 0
count_update = 0

################### sol ######################
lots = list(park.spaces.keys())
th = Worker(lots)
th.setDaemon(True)
th.start()
##############################################

if not cap.isOpened():
    raise IOError("Cannot open webcam")

while cap.isOpened():
    ret, origin = cap.read()
    curtime = time.time()
    key = cv2.waitKey(1)
    if ret:
        frame = cv2.resize(origin, (w, h))
        frame = warp.trans_img(frame,width=w,height=h)    #탑-다운뷰 변환
        img, point = yolo.run(**vars(opt),source=frame,model = model,device=device,stride=stride,names=names,imgsz=imgsz)

        if count_frame == 0:
            send_dict = park.set_occupied(point)
        elif count_frame % 10 == 0:
            park.is_occupied(point)
            send_dict = park.update_dict()
            count_update += 1
        if send_dict:
            ################### sol ######################
            for item in send_dict.items():
                th.update(item)
            ##############################################
        send_dict.clear()
        frame = park.draw_box(frame)
        sec = curtime - pretime
        pretime = curtime
        fps = 1/(sec)
        if count_frame % 5 == 0:
            str = "FPS : %0.1f" % fps
            
        #FPS 화면 표시
        cv2.putText(frame,str,(0,50),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),2)
        crop_img = frame[0:1080,0:1550]
        cv2.imshow(windows_name,crop_img)
        cv2.imshow(origin_name,origin)

        count_frame += 1
    
    if key & 0xFF == ord('s'):
        warp.save_H_mat(origin,width=w,height=h)
        points = warp.return_pts2()
        park = parking(points, n, m)
        park.divide_xy()    
    
    #영상 종료
    # if cap.get(cv2.CAP_PROP_FRAME_COUNT) != 0: # 저장된 영상일때
    #     if (cap.get(cv2.CAP_PROP_FRAME_COUNT) == cap.get(cv2.CAP_PROP_POS_FRAMES)) or (key & 0xFF == ord('e')):
    #         print("video_ close")
    #         break
    # else:
    #     if key & 0xFF == ord('e'):# 웹캠 일때
    #         print('webcam_close')
    #         break

cap.release()
cv2.destroyAllWindows()