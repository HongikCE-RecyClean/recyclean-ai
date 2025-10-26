from typing import Optional

from fastapi import APIRouter, UploadFile, File, HTTPException
from starlette.responses import JSONResponse
from tempfile import NamedTemporaryFile
import shutil
from ultralytics import YOLO

router = APIRouter()

# 모델 로딩
model = YOLO("../training/runs/detect/train/weights/best.pt")
CLASS_NAMES = model.names  # or ["glass", "can", "plastic", "battery"]

@router.post("/labeling")
async def labeling(trash_img: Optional[UploadFile] = File(None)):
    if trash_img is None or trash_img.filename == "":
        raise HTTPException(
            status_code=400,
            detail={
                "success": False,
                "error": "REQUEST_BODY_NULL_EXCEPTION",
                "message": "요청에 이미지 파일이 없습니다."
            }
        )

    try:
        # 임시 파일 저장
        with NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            shutil.copyfileobj(trash_img.file, tmp)
            tmp_path = tmp.name

        # 추론
        results = model(tmp_path, conf=0.1)[0]

        predictions = []
        for box in results.boxes:
            cls_id = int(box.cls[0].item())
            conf = round(box.conf[0].item(), 2)
            x1, y1, x2, y2 = map(lambda x: round(x.item(), 2), box.xyxy[0])

            predictions.append({
                "category": CLASS_NAMES[cls_id],
                "confidence": conf,
                "bbox": [x1, y1, x2, y2]
            })

        return JSONResponse(status_code=200, content={"predictions": predictions})

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "success": False,
                "error": "INTERNAL_SERVER_ERROR",
                "message": str(e)
            }
        )