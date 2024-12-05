# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the application source code to the container
COPY app_ver5.py /app/

# Copy any additional files required by the app (models, pickles, etc.)
COPY synonym_model.h5 /app/
COPY word_to_index.pkl /app/
COPY index_to_word.pkl /app/

# Install required system packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    make \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Expose the application's port
EXPOSE 5606

# Set the command to run the application
CMD ["python", "app_ver5.py"]
