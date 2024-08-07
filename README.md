# parking_availability
졸업 프로젝트/🚗특정 주차공간에 대한 주차 가능 여부 확인 앱   

👥**팀 구성**: 5인팀  
👩‍💻**역할**: 인공지능 모델(YOLOv5) 학습 및 주차 공간 탐색 기능 구현  
🧰**stacks**: ```python3``` ```pytorch``` ```openCV``` ```firebase``` ```flutter``` ```YOLOv5```

## 기획의도
CCTV 영상을 기반으로 이미지 탐색(detection) 모델을 학습시켜 탐색하고 결과를 사용자 앱에 표시해 사용자 편리를 추구함  
<img width="959" alt="fig1" src="https://github.com/noachilles/TIL/assets/74445032/0aa1267a-035a-4ed9-8139-cd108c2d0ff8">
fig 1. 설계 과정 이미지

## 제작 과정-딥러닝
### 데이터 수집
드론/웹캠으로 직접 수집 후 Roboflow 플랫폼으로 라벨링 작업 진행


### 데이터 전처리
CCTV 적용을 목적으로 OpenCV를 이용해 Grayscale, Blur, Noise가 첨가된 이미지를 생성  
총 2,600장의 이미지를 8:2(Train : Validation)로 구분해 학습에 활용  
(fig 2. 모델 학습 데이터)

### 딥러닝 모델 학습
NVIDIA V100 그래픽으로 학습 진행, 영상 속 실시간 객체 검출이 가능한 YOLOv5 모델을 활용  
![fig3](https://github.com/noachilles/TIL/assets/74445032/9a8daf32-97ae-4f89-8b96-83d93c6de66b)  
fig 3. 모델 학습 loss 그래프  
![fig4](https://github.com/noachilles/TIL/assets/74445032/015d75dd-df48-4df4-b469-7f2fa309ad55)  
fig 4. 모델 학습 예시 이미지

## 제작 과정-기능 구현  
### [실시간 검출](https://github.com/noachilles/parking_availability/blob/main/is_occupied.py)
학습된 모델을 활용해 영상 속 주차공간에 대한 주차 여부 파악해 색상으로 가시적으로 표시함  
<img width="391" alt="fig5" src="https://github.com/noachilles/TIL/assets/74445032/03294d92-904a-4e31-8b58-517159fc0321">  
fig 5. 실시간 차량 검출 이미지  
![fig6](https://github.com/noachilles/TIL/assets/74445032/0b01a2bc-348e-46ef-87fe-9ae20d395f8a)  
fig 6. 실시간 주차 여부 시각화 이미지  

### [Application 연동](https://github.com/noachilles/parking_availability/blob/main/parking_detect.py)
firebase server 연동을 통해 dictionary 형태로 데이터 전달
각 주차 공간의 번호를 사전 지정해두고 mapping
![fig7](https://github.com/noachilles/TIL/assets/74445032/6e211178-1bb6-4e82-b57e-3871c999ed24)  
fig 7. 실시간 주차 확인 앱 이미지