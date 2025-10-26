from ultralytics import YOLO

model = YOLO('yolov8n.pt')  # or yolov8s.pt, yolov8m.pt 등
model.train(data='recycle.yaml', epochs=150, imgsz=1280, batch=8, exist_ok=True)