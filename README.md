# 2023 DataMining

-----------------
![image](https://github.com/johaewon/openpose-code/assets/108321733/8931a96b-1768-49b8-8d02-f6327966412d)
![image](https://github.com/johaewon/openpose-code/assets/108321733/33cabae8-1c82-4f4c-bb09-72a68cf2c17d)

[**실시간 특정 영역 진입 시 얼굴 모자이크 해제 시스템**]


실시간 특정 영역 진입 시 얼굴 모자이크 해제 시스템은 딥러닝과 영상처리 기술을 사용하여 모자이크 처리 문제를 해결하는 시스템을 제안한다. 과거에는 모자이크 처리가 수동으로 이루어졌으나, 이 방식은 시간이 많이 걸리고 실수로 누락되는 경우가 발생할 수 있다. 이 시스템은 실시간으로 얼굴 모자이크를 자동으로 해제하는 인공지능 기반 CCTV 기술에 기반한다. OpenPose 모델을 사용하여 사람의 전신, 손, 얼굴, 발 등의 키포인트를 실시간으로 탐지한 후 영상 내 모든 사람에게 기본적으로 얼굴 모자이크 효과를 적용한다. 이후 OpenPose로 얻은 다리 영역의 확장된 박스와 특정 영역 사이의 IoU 값을 기준으로 모자이크 해제 이벤트를 결정한다. 이러한 시스템은 시시간 모자이크 처리를 자동화하여 사생활 침해 및 법적 문제를 예방하고, 효율적인 방송 제작을 가능하게 한다.

#### Paper
> https://www.koreascience.or.kr/article/CFKO201924664108405.page


#### 설명
실시간 특정 영역 진입 시 얼굴 모자이크 해제 시스템은 딥러닝과 영상처리 기술을 사용하여 모자이크 처리 문제를 해결하는 시스템을 제안한다. 과거에는 모자이크 처리가 수동으로 이루어졌으나, 이 방식은 시간이 많이 걸리고 실수로 누락되는 경우가 발생할 수 있다. 이 시스템은 실시간으로 얼굴 모자이크를 자동으로 해제하는 인공지능 기반 CCTV 기술에 기반한다. OpenPose 모델을 사용하여 사람의 전신, 손, 얼굴, 발 등의 키포인트를 실시간으로 탐지한 후 영상 내 모든 사람에게 기본적으로 얼굴 모자이크 효과를 적용한다. 이후 OpenPose로 얻은 다리 영역의 확장된 박스와 특정 영역 사이의 IoU 값을 기준으로 모자이크 해제 이벤트를 결정한다. 이러한 시스템은 시시간 모자이크 처리를 자동화하여 사생활 침해 및 법적 문제를 예방하고, 효율적인 방송 제작을 가능하게 한다.

#### 정리
* 실시간으로 특정 객체에 이미지 처리를 수행하는 시스템 입니다.   
* 촬영 후 편집으로 수행했던 작업들을 실시간으로 처리할 수 있으므로 실시간 미디어 매체에 활용이 가능합니다.   
* 임베디드 보드(Jetson tx1 board)에서 동작할 수 있습니다.   
* `YOLO`와 `FaceNet`을 활용하여 구현하였습니다.


***

### 특정 인물을 제외한 모든 인물 모자이크

<p align="center">
<img src="./image/option_1_test.gif" alt="option_1"/>
</p>

***

### 사물 모자이크

<p align="center">
<img src="./image/option_2_test.gif" alt="option_2"/>
</p>

***

### 임베디드 보드 테스트 (Jetson TX1 board)

<p align="center">
  <img src="./image/JTX1_devkit.png" alt="JTX1_devkit" style="width:500px;"/>
<img src="./image/board_test_image.png" alt="board_test_image" style="width:500px;"/>
<img src="./image/board_test.gif" alt="board_test"/>
</p>

## Run
> ⛔ github 정책으로 인해 100MB 이상의 weight file은 포함되지 않았습니다. (weight file 없이 실행 불가) ⛔ 
#### 사물 모자이크
```
# step 1
./cap.bash object
```

#### 특정 인물 제외 모든 인물 모자이크
```
# step 1. 특정 인물 얼굴 캡쳐
./cap.bash crop

# step 2. FaceNet 학습
./cap.bash train

# step 3. 실행
./cap.bash face
```


## 세부 구현 내용
#### YOLO-FaceNet 통신

<p align="center">
<img src="./image/shared_memory.png" alt="model_connection" style="width:500px;"/>
</p>

YOLO는 `C`, FaceNet은 `Python`으로 구현되어 있으므로 입출력값을 공유하기 위한 공유 메모리를 사용합니다.

#### MeanShift

병목 현상 방지를 위해 30fps 중 1fps는 FaceNet, 29fps는 MeanShift 알고리즘을 사용하였습니다.

#### 임베디드 환경

YOLO(Tiny Yolo)는 임베디드 보드로 수행, 30fps 마다 PC와 Socket 통신하여 연산을 요청, FaceNet의 결과를 제공받습니다.


## Development Environment
#### OS   
* Linux   
#### Language
* C   
* Python   
#### GPU   
|PC|TX1|
|---|---|
|NVIDIA CUDA® Cores-4352|256-core NVIDIA Maxwell™|

## Reference
#### Site
* [YOLOv3](https://pjreddie.com/darknet/yolo/)
* [Darknet YOLO분석](https://pgmrlsh.tistory.com/5?category=766787)
#### Paper
* [YOLOv3](https://arxiv.org/pdf/1804.02767.pdf)
* [FaceNet](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Schroff_FaceNet_A_Unified_2015_CVPR_paper.pdf)
#### Code
* https://github.com/pjreddie/darknet
* https://github.com/davidsandberg/facenet
* https://github.com/shanren7/real_time_face_recognition
* https://github.com/msindev/Facial-Recognition-Using-FaceNet-Siamese-One-Shot-Learning

