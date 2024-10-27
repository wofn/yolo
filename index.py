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
image_path = '/Users/sin-yeonghyeon/Desktop/test_img/test7.jpeg'
img = cv2.imread(image_path)

# CLAHE 적용 함수
def apply_clahe_y_channel(image):
    ycrcb_array = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb_array)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    y_clahe = clahe.apply(y)
    merge_array = cv2.merge([y_clahe, cr, cb])
    return cv2.cvtColor(merge_array, cv2.COLOR_YCrCb2BGR)

# YOLO 객체 인식 함수
def detect_objects(image):
    blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    confidences = []
    boxes = []
    class_ids = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    return confidences, boxes, class_ids, indexes

# 원본 이미지로 객체 인식
height, width, channels = img.shape
confidences, boxes, class_ids, indexes = detect_objects(img)

# 최고 신뢰도가 0.7 미만인 경우 이미지 전처리 적용
is_transformed = False
if len(confidences) == 0 or max(confidences) < 0.8:
    print("Adjusting image for better detection...")
    img = apply_clahe_y_channel(img)
    is_transformed = True
    confidences, boxes, class_ids, indexes = detect_objects(img)

# 경계 상자 및 클래스 이름 그리기
colors = {cls: (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for cls in classes}

if len(indexes) > 0:
    for i in indexes.flatten():
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        confidence = confidences[i]
        color = colors[label]
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        cv2.putText(img, f"{label} {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# YOLO 결과 이미지 저장 경로
yolo_result_dir = '/Users/sin-yeonghyeon/Desktop/3-2/yolo_test/yolo_results'
os.makedirs(yolo_result_dir, exist_ok=True)  # 디렉토리가 없으면 생성
yolo_result_image_path = os.path.join(yolo_result_dir, 'result_with_boxes.jpg')
cv2.imwrite(yolo_result_image_path, img)

# 변환 여부에 따른 제목 설정
title = "변환된 이미지" if is_transformed else "원본 이미지"

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
    <h1>YOLO Image Analysis Result - {title}</h1>
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
