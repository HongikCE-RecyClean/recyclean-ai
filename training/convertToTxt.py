import json
import os

# 클래스 매핑 (한글 -> 영문 -> YOLO class_id)
class_name_map = {
    "금속캔" : "can",
    "종이" : "paper",
    "페트병" : "pet",
    "플라스틱" : "plastic",
    "스티로폼" : "styrofoam",
    "비닐" : "vinyl",
    "유리병" : "glass",
    "건전지" : "battery",
    "형광등" : "fluorescent_tube"
}

class_id_map = {
    "can" : 0,
    "paper" : 1,
    "pet" : 2,
    "plastic" : 3,
    "styrofoam" : 4,
    "vinyl" : 5,
    "glass" : 6,
    "battery" : 7,
    "fluorescent_tube" : 8
}

input_json_dir = "json/val"
output_label_dir = "dataset/val/labels"

def convert_to_yolo(json_data):
    image_info = json_data["IMAGE_INFO"]
    annotations = json_data["ANNOTATION_INFO"]

    img_w = image_info["IMAGE_WIDTH"]
    img_h = image_info["IMAGE_HEIGHT"]
    filename = image_info["FILE_NAME"]
    name, _ = os.path.splitext(filename)

    yolo_lines = []

    for ann in annotations:
        class_name_kor = ann["CLASS"]
        class_name_eng = class_name_map.get(class_name_kor)
        class_id = class_id_map.get(class_name_eng)

        if class_id is None:
            continue

        points = ann["POINTS"]

        if len(points[0]) == 4:
            # 박스형
            x_min, y_min, box_w, box_h = points[0]

        elif len(points[0]) == 2:
            # 다각형형 → bounding box로 변환
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)
            box_w = x_max - x_min
            box_h = y_max - y_min

        else:
            print(f"Unknown POINTS format in {filename}: {points}")
            continue

        # 절대좌표 → YOLO 좌표계 변환
        x_center = (x_min + box_w / 2) / img_w
        y_center = (y_min + box_h / 2) / img_h
        width    = box_w / img_w
        height   = box_h / img_h

        yolo_line = f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
        yolo_lines.append(yolo_line)

    return name, yolo_lines

def save_yolo_labels(name, lines, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, f"{name}.txt"), "w") as f:
        for line in lines:
            f.write(line + "\n")


for filename in os.listdir(input_json_dir):
    if filename.endswith(".json"):
        with open(os.path.join(input_json_dir, filename), "r", encoding="utf-8") as f:
            data = json.load(f)
        name, lines = convert_to_yolo(data)
        save_yolo_labels(name, lines, output_label_dir)