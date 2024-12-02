from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
import csv
app = Flask(__name__)
import json

# Định nghĩa số lượng thể loại
num_genres = 3862  # Số lượng thể loại hiện tại (theo số cột trong model)

# Load mô hình CNN đã được train
model = tf.keras.models.load_model('video_recommendation_model_second_ver5.h5')

# Load dữ liệu từ file Vocabulary.csv
genre_map = {}
index_map = {}

with open('vocabulary.sql', newline='', encoding='utf-8') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)  # Bỏ qua header
    for row in reader:
        index = int(row[0])
        genre = row[1]
        genre_map[genre.lower()] = index  # Ánh xạ tên thể loại sang index
        index_map[index] = genre  # Ánh xạ index sang tên thể loại

# Hàm ánh xạ tên thể loại sang index
def genre_to_index(genre):
    return genre_map.get(genre.lower(), -1)  # Nếu không tìm thấy, trả về -1

# Hàm ánh xạ index sang tên thể loại
def index_to_genre(index):
    return index_map.get(index, "Unknown")  # Nếu không tìm thấy, trả về "Unknown"

# Hàm tạo embedding vector từ input
def create_input_vector(genres_watched):
    input_vector = np.zeros((1, 1152))  # Tạo vector đầu vào có kích thước 3862
    for genre, count in genres_watched.items():
        genre_idx = genre_to_index(genre)
        if genre_idx != -1:  # Nếu thể loại tồn tại
            input_vector[0, genre_idx] = count
    input_vector = np.expand_dims(input_vector, axis=-1)  # Thêm chiều cho CNN (batch, num_genres, 1)
    return input_vector

# API nhận số lần xem các thể loại và trả về các thể loại được đề xuất
@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.json
    genres_watched = data['genres_watched']  # Nhận dữ liệu dạng: {"Game": 3, "Musician": 2}
    
    # Tạo input vector từ số lần xem các thể loại
    input_vector = create_input_vector(genres_watched)
    
    # Dự đoán từ mô hình CNN
    predicted_genres = model.predict(input_vector)
    
    # Lấy top 5 thể loại được đề xuất (dưới dạng chỉ số)
    recommended_genres_idx = np.argsort(predicted_genres[0])[-10:]
    
    # Ánh xạ từ chỉ số sang tên thể loại
    recommended_genres = [index_to_genre(idx) for idx in recommended_genres_idx]
    
    return jsonify({'recommended_genres': recommended_genres})





from keybert import KeyBERT
kw_model = KeyBERT()
from kafka import KafkaConsumer, KafkaProducer

@app.route('/extract_keywords', methods=['POST'])
def extract_keywords():
    data = request.json
    description = data.get("description", "")
    if not description:
        return jsonify({"error": "Description is required"}), 400

    keywords = kw_model.extract_keywords(description, keyphrase_ngram_range=(1, 2), stop_words='english')
    keyword_list = [keyword[0] for keyword in keywords]
    
    return jsonify({"keywords": keyword_list})

# Kafka Consumer Configuration
consumer = KafkaConsumer(
    'video-description-topic',
    bootstrap_servers=['localhost:9092'],
    auto_offset_reset='earliest',
    enable_auto_commit=True,
    group_id='video-description-group',
    value_deserializer=lambda x: json.loads(x.decode('utf-8'))
)

# Kafka Producer Configuration
producer = KafkaProducer(
    bootstrap_servers=['localhost:9092'],
    value_serializer=lambda x: json.dumps(x).encode('utf-8')
)


import threading
def consume_kafka():
    for message in consumer:
        data = message.value
        video_id = data.get("videoID", "")
        description = data.get("description", "")
        if description:
            print(f"Processing description: {description}")
            # Gọi hàm phân tích keywords tại đây
            keywords = kw_model.extract_keywords(description, keyphrase_ngram_range=(1, 2), stop_words='english')
            keyword_list = [keyword[0] for keyword in keywords]
            print(f"Extracted keywords: {keyword_list}")

            producer.send(
                'video-genres-topic',
                value={"videoID": video_id, "genres": keyword_list}
            )
            print(f"Sent keywords to Kafka: {keyword_list}")

if __name__ == '__main__':
    kafka_thread = threading.Thread(target=consume_kafka)
    kafka_thread.start()
    app.run(host='0.0.0.0', port=5050)