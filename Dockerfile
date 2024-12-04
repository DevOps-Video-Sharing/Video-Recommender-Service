# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container
COPY . /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir \
    flask \
    kafka-python \
    tensorflow \
    scikit-learn \
    numpy \
    keybert \
    pickle5

# Expose the port the app runs on
EXPOSE 5606

# Run the application
CMD ["python", "app_ver5.py"]
