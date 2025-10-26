from fastapi import FastAPI
from api.routes import router as api_router
import logging
from fastapi import FastAPI, Request
import time
import os

# 로그 폴더 자동 생성
os.makedirs("logs", exist_ok=True)

# 로거 설정
logger = logging.getLogger("api_logger")
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler("logs/api.log", encoding="utf-8")
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

app = FastAPI()

app.include_router(api_router, prefix = "/api")

# 요청 로그 미들웨어
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = (time.time() - start_time) * 1000
    client_host = request.client.host
    method = request.method
    url = request.url.path
    status_code = response.status_code

    logger.info(f"{client_host} {method} {url} {status_code} {process_time:.2f}ms")
    return response