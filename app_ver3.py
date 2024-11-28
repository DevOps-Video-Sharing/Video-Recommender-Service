from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
import csv

app = Flask(__name__)

# Định nghĩa số lượng thể loại
num_genres = 3862  # Số lượng thể loại hiện tại (theo số cột trong model)

# Giả sử mô hình đã được train với embedding 128
embedding_size = 128

# Load mô hình đã train
model = tf.keras.models.load_model('video_recommendation_model_ver3.h5')

# Load dữ liệu từ file Vocabulary.csv
genre_map = {}
index_map = {}

with open('vocabulary.csv', newline='', encoding='utf-8') as csvfile:
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
    input_vector = np.zeros((1, num_genres))  # Tạo vector đầu vào có kích thước 3862
    for genre, count in genres_watched.items():
        genre_idx = genre_to_index(genre)
        if genre_idx != -1:  # Nếu thể loại tồn tại
            input_vector[0, genre_idx] = count
    return input_vector  # Trả về vector có kích thước đầy đủ (1, 3862)

# API nhận số lần xem các thể loại và trả về các thể loại được đề xuất
@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.json
    genres_watched = data['genres_watched']  # Nhận dữ liệu dạng: {"Game": 3, "Musician": 2}
    
    # Tạo input vector từ số lần xem các thể loại
    input_vector = create_input_vector(genres_watched)
    
    # Dự đoán từ mô hình AI
    predicted_genres = model.predict(input_vector)
    
    # Lấy top 5 thể loại được đề xuất (dưới dạng chỉ số)
    recommended_genres_idx = np.argsort(predicted_genres[0])[-10:]
    
    # Ánh xạ từ chỉ số sang tên thể loại
    recommended_genres = [index_to_genre(idx) for idx in recommended_genres_idx]
    
    return jsonify({'recommended_genres': recommended_genres})


if __name__ == '__main__':
    app.run(debug=True, port=5050)
