# ใช้ Python base image
FROM python:3.10-slim

# สร้าง working directory
WORKDIR /app

# คัดลอกไฟล์ทั้งหมดเข้า image
COPY . .

# ติดตั้ง dependency
RUN pip install --no-cache-dir -r requirements.txt

# รันระบบ auto trade
CMD ["python", "auto_trade_loop.py"]