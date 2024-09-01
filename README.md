# parking_availability
졸업 프로젝트/🚗특정 주차공간에 대한 주차 가능 여부 확인 앱   

🤝**팀 구성**: 5인팀  
👩‍💻**역할**: 인공지능 모델(YOLOv5) 학습 및 주차 공간 탐색 기능 구현  
🧰**stacks**: ```Python3``` ```Pytorch``` ```OpenCV``` ```firebase``` ```flutter``` ```YOLOv5```

## 기획의도
* CCTV 영상을 기반으로 이미지 탐색(detection) 모델을 학습시켜 탐색하고 결과를 사용자 앱에 표시해 사용자 편의를 추구함  
<img width="959" alt="fig1" src="https://github.com/noachilles/TIL/assets/74445032/0aa1267a-035a-4ed9-8139-cd108c2d0ff8">
<div align="center">fig 1. 계획서 일부 이미지</div>

## 제작 과정-딥러닝
### 데이터 수집
* 다양한 source로부터 데이터를 수집함  
    1. 웹캠, 드론으로 직접 촬영  
    2. 웹 크롤링 결과를 바탕으로 데이터 선별  
    3. Computer Vision 학습데이터 COCO dataset 활용  

* 이후 Roboflow 플랫폼을 활용해 직접 데이터 라벨링 작업 진행  


### 데이터 전처리
* CCTV 적용을 목적으로 OpenCV를 이용해 Grayscale, Blur, Noise가 첨가된 이미지를 생성  
* 기존 1400장의 이미지<span style="color: #777777">(Number of Annotations: 17918)</span>를 8:2<span style="color: #777777">(Train : Validation)</span>로 구분해 학습에 활용   

✅데이터세트 확인 [Dataset Link](https://app.roboflow.com/rclab-jyxhq/coco-drone/health)  

<img src="https://github.com/user-attachments/assets/7e327ff2-c1e3-4c02-9875-772c54f08519">  
<div align="center">fig 2. 학습 데이터 생성 과정</div>



### 딥러닝 모델 학습
* **개발환경**: LINUX, NVIDIA V100 
* 영상 속 실시간 객체 검출이 가능한 YOLOv5 모델 사용  
<img src="https://github.com/noachilles/TIL/assets/74445032/9a8daf32-97ae-4f89-8b96-83d93c6de66b" width="100%" />
<div align="center">fig 3. 모델 학습 과정 그래프</div>  
<div align="center"><img src="https://github.com/noachilles/TIL/assets/74445032/015d75dd-df48-4df4-b469-7f2fa309ad55" width=80%></div>  
<div align="center">  fig 4. 모델 학습 예시 이미지</div>

## 제작 과정-기능 구현 및 앱  
### OpenCV   
* CCTV 시야를 고려해 설계했으므로 CCTV의 각도에서 차량이 중복되거나 겹치는 경우를 고려해 OpenCV 모듈을 사용해 영상을 top-down view로 왜곡하고자 함

<img src="https://github.com/user-attachments/assets/e4435a33-7b29-4ab8-a864-5c4da81a8ef8" width=50%/><img src="https://github.com/user-attachments/assets/8c3283b3-1e73-4de7-b657-386b1b1f527a" width=50%/>  
<div align="center">fig 5. 좌, 우 각각 top-down view 적용 전, 후 사진</div>

### [실시간 검출](https://github.com/noachilles/parking_availability/blob/main/is_occupied.py)
* 학습된 모델을 활용해 영상 속 주차공간에 대한 주차 여부 파악해 색상으로 가시적으로 표시함  
<div align="center"><img src="https://github.com/noachilles/TIL/assets/74445032/03294d92-904a-4e31-8b58-517159fc0321" width=70%/></div>  
<div align="center">fig 5. 실시간 주차구역 차량 검출 이미지</div>

<div align="center"><img src="https://github.com/noachilles/TIL/assets/74445032/0b01a2bc-348e-46ef-87fe-9ae20d395f8a" width=70%/></div>  
<div align="center">fig 6. 실시간 주차 여부 시각화 이미지</div>


### [Application 연동](https://github.com/noachilles/parking_availability/blob/main/parking_detect.py)  
* firebase server 연동을 통해 dictionary 형태로 데이터 전달  
* Figma 활용해 애플리케이션 디자인 설계   
* Flutter 활용해 애플리케이션 구현
<div align="center"><img src="https://github.com/noachilles/TIL/assets/74445032/6e211178-1bb6-4e82-b57e-3871c999ed24"/></div>
<div align="center">fig 7. 실시간 주차 확인 앱 이미지</div>