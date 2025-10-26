import cv2
from pathlib import Path

img_path = Path("../../training/dataset/train/images/test.jpg")
label_path = Path("../../training/dataset/train/labels/test.txt")
class_names =   [
    "can",
    "paper",
    "pet",
    "plastic",
    "styrofoam",
    "vinyl",
    "glass",
    "battery",
    "fluorescent_tube"
  ]
# 이미지 불러오기
img = cv2.imread(str(img_path))
h, w = img.shape[:2]
print(f"시각화에 사용된 실제 이미지 해상도: {w} x {h}")

# 라벨 그리기
with open(label_path, "r") as f:
    lines = f.readlines()

for line in lines:
    parts = line.strip().split()
    cls_id = int(parts[0])
    cx, cy, bw, bh = map(float, parts[1:])

    # YOLO → pixel 좌표 변환
    x1 = int((cx - bw / 2) * w)
    y1 = int((cy - bh / 2) * h)
    x2 = int((cx + bw / 2) * w)
    y2 = int((cy + bh / 2) * h)

    # 박스 및 라벨 그리기
    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)  # 파란 박스
    cv2.putText(img, class_names[cls_id], (x1, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

# 결과 저장
out_path = img_path.parent / f"{img_path.stem}_gt.jpg"
cv2.imwrite(str(out_path), img)
print(f"GT 시각화 이미지 저장 완료: {out_path}")