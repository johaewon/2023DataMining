import torch
import cv2
import numpy as np
import datetime
from shapely.geometry import Polygon


# MPII에서 각 파트 번호, 선으로 연결될 POSE_PAIRS
BODY_PARTS = { "Head": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
                "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
                "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "Chest": 14,
                "Background": 15 }

POSE_PAIRS = [ ["Head", "Neck"], ["Neck", "RShoulder"], ["RShoulder", "RElbow"],
                ["RElbow", "RWrist"], ["Neck", "LShoulder"], ["LShoulder", "LElbow"],
                ["LElbow", "LWrist"], ["Neck", "Chest"], ["Chest", "RHip"], ["RHip", "RKnee"],
                ["RKnee", "RAnkle"], ["Chest", "LHip"], ["LHip", "LKnee"], ["LKnee", "LAnkle"] ]

print("바디파츠 머리 : " , BODY_PARTS)
# 각 파일 path
protoFile = "./openpose-master/models/pose/mpi/pose_deploy_linevec_faster_4_stages.prototxt"
weightsFile = "./openpose-master/models/pose/mpi/pose_iter_160000.caffemodel"


intrusion_area = Polygon([
         (653,451),
          (824,507),
           (625,718),
            (380,718),
            (449,664),
            (438, 604)])

# 위의 path에 있는 network 불러오기
net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)


fourcc = cv2.VideoWriter_fourcc(*'XVID')
video_path = "./sample_video/sample.mp4"

cap = cv2.VideoCapture(video_path)

video_w, video_h, video_fps = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), float(cap.get(cv2.CAP_PROP_FPS))
video = cv2.VideoWriter("./result/" + "openpose_result.avi", fourcc, video_fps, (video_w, video_h))


inputWidth=320;
inputHeight=240;
inputScale=1.0/255;


def calculate_iou(box1, box2):
    # 교차 영역(Intersection)의 면적 계산
    inter_area = box1.intersection(box2).area
    # 합집합 영역(Union)의 면적 계산
    union_area = box1.area + box2.area - inter_area
    # IoU 계산
    iou = inter_area / union_area if union_area != 0 else 0
    return iou


    #반복문을 통해 카메라에서 프레임을 지속적으로 받아옴
while True:  #아무 키나 누르면 끝난다.
   
    hasFrame, frame = cap.read()  
    
    frameWidth = frame.shape[1]
    frameHeight = frame.shape[0]
    
    inpBlob = cv2.dnn.blobFromImage(frame, inputScale, (inputWidth, inputHeight), (0, 0, 0), swapRB=False, crop=False)
    
    imgb=cv2.dnn.imagesFromBlob(inpBlob)
    
    intrusion_area_coords = intrusion_area.exterior.coords
    intrusion_area_coords = [(int(x), int(y)) for x, y in intrusion_area_coords]
    cv2.polylines(frame, [np.array(intrusion_area_coords)], isClosed=True, color=(0, 0, 255), thickness=2)
        
    # network에 넣어주기
    net.setInput(inpBlob)

    # 결과 받아오기
    output = net.forward()
    print(0)

    # 키포인트 검출시 이미지에 그려줌
    points = []
    for i in range(0,15):
        # 해당 신체부위 신뢰도 얻음.
        probMap = output[0, i, :, :]
        
        # global 최대값 찾기
        minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)

        # 원래 이미지에 맞게 점 위치 변경
        x = (frameWidth * point[0]) / output.shape[3]
        y = (frameHeight * point[1]) / output.shape[2]

        # 키포인트 검출한 결과가 0.1보다 크면(검출한곳이 위 BODY_PARTS랑 맞는 부위면) points에 추가, 검출했는데 부위가 없으면 None으로    
        if prob > 0.1 :    
            cv2.circle(frame, (int(x), int(y)), 3, (0, 255, 255), thickness=-1, lineType=cv2.FILLED) # circle(그릴곳, 원의 중심, 반지름, 색)
            cv2.putText(frame, "{}".format(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, lineType=cv2.LINE_AA)
            points.append((int(x), int(y)))
        else :
            points.append(None)
        

    # 각 POSE_PAIRS별로 선 그어줌 (머리 - 목, 목 - 왼쪽어깨, ...)
    for pair in POSE_PAIRS:
        partA = pair[0]             # Head
        print(partA)
        partA = BODY_PARTS[partA]   # 0
        print(partA)
        partB = pair[1]             # Neck
        partB = BODY_PARTS[partB]   # 1
            
        #partA와 partB 사이에 선을 그어줌 (cv2.line)
        if points[partA] and points[partB]:
            print("1 : ", points[partA])
            cv2.line(frame, points[partA], points[partB], (0, 255, 0), 2)

            if partA == BODY_PARTS["Head"] :
                (a, b) = points[partA]
                (m, n) = points[partB]
                qu = abs(n - b)
                x = max(a - (qu // 2), 0)
                y = max(b, 0)
                w = min(qu, frame.shape[1] - x)
                h = min(qu, frame.shape[0] - y)

                roi = frame[y:y+h, x:x+w]

                factor = 10
                small_roi = cv2.resize(roi, (w // factor, h // factor))
                mosaic_roi = cv2.resize(small_roi, (w, h), interpolation=cv2.INTER_NEAREST)
                mosaic_area = Polygon([(x, y), (x + w, y), (x + w, y + h), (x, y + h)])

                frame[y:y+h, x:x+w] = mosaic_roi

            if partA == BODY_PARTS["RKnee"] or partA == BODY_PARTS["LKnee"]:
                # 다리에 해당하는 키포인트가 모두 있는지 확인
                if all([points[BODY_PARTS["RKnee"]], points[BODY_PARTS["LKnee"]],
                        points[BODY_PARTS["LAnkle"]], points[BODY_PARTS["RAnkle"]]]):
                    knee_area = Polygon([
                            points[BODY_PARTS["RKnee"]],
                            points[BODY_PARTS["LKnee"]],
                            points[BODY_PARTS["LAnkle"]],
                            points[BODY_PARTS["RAnkle"]]])
                    
                    knee_area_coords = knee_area.exterior.coords
                    knee_area_coords = [(int(x), int(y)) for x, y in knee_area_coords]
                    cv2.polylines(frame, [np.array(knee_area_coords)], isClosed=True, color=(255, 0, 0), thickness=2)

            iou = calculate_iou(mosaic_area, knee_area)
                    
        
   

                
    cv2.imshow("Output-Keypoints",frame)
    video.write(frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
            
        break



video.release()  #카메라 장치에서 받아온 메모리 해제
cv2.destroyAllWindows() #모든 윈도우 창 닫음
