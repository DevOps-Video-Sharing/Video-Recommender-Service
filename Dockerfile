# Sử dụng Python 3.9 làm image cơ bản
FROM python:3.9-slim

# Thiết lập thư mục làm việc bên trong container
WORKDIR /app

# Sao chép file requirements.txt vào container
COPY requirements.txt /app/requirements.txt

# Cài đặt các thư viện cần thiết
RUN pip install --no-cache-dir -r requirements.txt

# Sao chép toàn bộ mã nguồn ứng dụng vào container
COPY . /app

# Mở cổng 5606
EXPOSE 5606

# Chạy ứng dụng
CMD ["python", "app.py"]
