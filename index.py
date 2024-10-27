import cv2
import numpy as np
import os
import random

# YOLO 모델 불러오기
net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')

# COCO 데이터셋 클래스 이름 불러오기
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# YOLO 네트워크의 출력 레이어 이름 가져오기
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

# 이미지 불러오기
image_path = '/Users/sin-yeonghyeon/Desktop/test_img/test2.jpg'
img = cv2.imread(image_path)
height, width, channels = img.shape

# 이미지 전처리
blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
net.setInput(blob)
outs = net.forward(output_layers)

# 탐지된 객체 정보 저장
class_ids = []
confidences = []
boxes = []

# 결과 해석
for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5:
            # 경계 상자 계산
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)

            x = int(center_x - w / 2)
            y = int(center_y - h / 2)

            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

# 비최대 억제 적용
indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

# 클래스별 색상 매핑
colors = {cls: (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for cls in classes}

# 경계 상자 및 클래스 이름 그리기
for i in range(len(boxes)):
    if i in indexes:
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        confidence = confidences[i]
        color = colors[label]  # 클래스에 따른 색상 선택
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        cv2.putText(img, f"{label} {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# YOLO 결과 이미지 저장 경로
yolo_result_dir = '/Users/sin-yeonghyeon/Desktop/3-2/yolo_test/yolo_results'
os.makedirs(yolo_result_dir, exist_ok=True)  # 디렉토리가 없으면 생성
yolo_result_image_path = os.path.join(yolo_result_dir, 'result_with_boxes.jpg')
cv2.imwrite(yolo_result_image_path, img)

# YOLO 결과 HTML 파일 생성
yolo_html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>YOLO Image Analysis Result</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        img {{ max-width: 100%; height: auto; border: 2px solid #333; }}
    </style>
</head>
<body>
    <h1>YOLO Image Analysis Result</h1>
    <img src="{yolo_result_image_path}" alt="Analyzed Image">
    <h2>Detected Objects:</h2>
    <ul>
"""

if len(indexes) == 0:
    yolo_html_content += "<li>No objects detected.</li>"
else:
    for i in indexes.flatten():
        label = str(classes[class_ids[i]])
        confidence = confidences[i]
        yolo_html_content += f"<li>{label} with confidence {confidence:.2f}</li>"

yolo_html_content += """
    </ul>
</body>
</html>
"""

# YOLO 결과 HTML 파일 저장 경로
yolo_html_file_path = os.path.join(yolo_result_dir, 'yolo_image_analysis_result.html')
with open(yolo_html_file_path, "w") as html_file:
    html_file.write(yolo_html_content)

print(f"Results saved to {yolo_result_image_path} and {yolo_html_file_path}")
