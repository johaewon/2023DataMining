# 2023 DataMining

-----------------
실행 사진 

![image](https://github.com/johaewon/openpose-code/assets/108321733/8931a96b-1768-49b8-8d02-f6327966412d)
![image](https://github.com/johaewon/openpose-code/assets/108321733/33cabae8-1c82-4f4c-bb09-72a68cf2c17d)

[**실시간 특정 영역 진입 시 얼굴 모자이크 해제 시스템**]


#### Paper
> [214583_조혜원_Final_Report.docx](https://github.com/johaewon/2023DataMining/files/13696543/214583_._Final_Report.docx)



#### 설명
실시간 특정 영역 진입 시 얼굴 모자이크 해제 시스템은 딥러닝과 영상처리 기술을 사용하여 모자이크 처리 문제를 해결하는 시스템을 제안한다. 과거에는 모자이크 처리가 수동으로 이루어졌으나, 이 방식은 시간이 많이 걸리고 실수로 누락되는 경우가 발생할 수 있다. 이 시스템은 실시간으로 얼굴 모자이크를 자동으로 해제하는 인공지능 기반 CCTV 기술에 기반한다. OpenPose 모델을 사용하여 사람의 전신, 손, 얼굴, 발 등의 키포인트를 실시간으로 탐지한 후 영상 내 모든 사람에게 기본적으로 얼굴 모자이크 효과를 적용한다. 이후 OpenPose로 얻은 다리 영역의 확장된 박스와 특정 영역 사이의 IoU 값을 기준으로 모자이크 해제 이벤트를 결정한다. 이러한 시스템은 시시간 모자이크 처리를 자동화하여 사생활 침해 및 법적 문제를 예방하고, 효율적인 방송 제작을 가능하게 한다.




## Run


### OpenPose 실행
```
https://github.com/CMU-Perceptual-Computing-Lab/openpose
```
위 링크에 들어가 openpose를 실행하기 위한 설치 방법을 따르기.

### 실행
```
# python
python pose_webcam.py
```

### 사용된 video, 결과물
'''
사용된 비디오와 결과물은
sample_vido 폴더와
result 폴더에 저장되어 있습니다.
'''

## 필요한 패키지

* torch
* opencv-python
* numpy
* shapely

## Development Environment
#### Window
* Cuda 11.8
* python == 3.10.13
* pytorch == 2.0.0
* torchvision == 0.15.0

