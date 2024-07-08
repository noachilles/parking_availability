from distutils.util import change_root
from pprint import pprint
import cv2
import copy

class parking:

    # 각 주차공간의 고유번호를 지정하고 False로 초기화
    def __init__(self, point_lst, nx=1, ny=1):
        self.spaces = {'180758':False, '152737':False, '152750':False, '152801':False, '152811':False, '152822':False, '152834':False, '152839':False, '152846':False, '152858':False, '152919':False, '152929':False, '152940':False, '152946':False, '152952':False, '152959':False, '153007':False, '153018':False}
        self.point_lst = point_lst
        self.list_l = []
        self.ocp_list = []
        self.nx = nx
        self.ny = ny
        self.alpha = 0.4    
    
    # 전체 주차공간을 가로 nx개, 세로 ny개로의 개별 주차공간으로 나눔
    def divide_xy(self):
        x1 = self.point_lst[0][0]
        x2 = self.point_lst[1][0]
        y1 = self.point_lst[0][1]
        y2 = self.point_lst[1][1]
        dx = (x2 - x1)/self.nx
        dy = (y2 - y1)/self.ny
        num = 0
        for i in range(0, self.ny, 1):
            for j in range(0, 2*self.nx, 2):
                key = list(self.spaces.keys())[num]
                self.spaces[key] = [False, (int(x1 + dx * (j/2)), int(y1 + dy * i)), (int(x1 + dx * (j/2+1)), int(y1 + dy * (i+1))), 0]
                num += 1
                
    # 개별 주차공간 영역에 주차된 차가 있으면 빨간색, 없으면 초록색 사각형으로 표시함
    def draw_box(self, img):
        im = copy.deepcopy(img)
        for key in self.spaces.keys():
            space = self.spaces[key]
            if space[0]:
                cv2.rectangle(im, space[1], space[2], (0, 0, 200), -1)
    
            else:
                cv2.rectangle(im, space[1], space[2], (0, 200, 0), -1)

        img = cv2.addWeighted(img, self.alpha, im, 1 - self.alpha, 0)
        return img
    
    # 영상이 시작될 때 주차공간별 차량 유무 확인 후 update(초기화)
    def set_occupied(self, points):
        send_dict = dict()
        for key in self.spaces.keys():
            ocp = False
            x1 = self.spaces[key][1][0]
            x2 = self.spaces[key][2][0]
            y1 = self.spaces[key][1][1]
            y2 = self.spaces[key][2][1]
            for i in range(len(points)):
                if (points[i][0] > x1 and points[i][0] < x2) and (points[i][1] > y1 and points[i][1] < y2):
                    ocp = True
            if (self.spaces[key][0] != ocp):
                self.spaces[key][0] = ocp
            send_dict[key] = ocp
        return send_dict
        
    # 영상에서 주차공간별 차량 유무 확인 후 update
    def is_occupied(self, points):
        for key in self.spaces.keys():
            ocp = False
            x1 = self.spaces[key][1][0]
            x2 = self.spaces[key][2][0]
            y1 = self.spaces[key][1][1]
            y2 = self.spaces[key][2][1]
            for i in range(len(points)):
                if (points[i][0] > x1 and points[i][0] < x2) and (points[i][1] > y1 and points[i][1] < y2):
                    ocp = True
            if (self.spaces[key][0] != ocp):
                self.spaces[key][3] += 1

    # firebase-application 연동된 서버에 전송할 dictionary update
    def update_dict(self):
        send_dict = dict()
        for key in self.spaces.keys():
            if self.spaces[key][3] >= 8:
                self.spaces[key][0] = not self.spaces[key][0]
                send_dict[key] = self.spaces[key][0]
                self.spaces[key][3] = 0
        return send_dict