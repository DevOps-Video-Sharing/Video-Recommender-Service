from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf

app = Flask(__name__)

# Định nghĩa số lượng thể loại
num_genres = 3862  # Số lượng thể loại hiện tại

# Giả sử mô hình đã được train với embedding 128
embedding_size = 128

# Load mô hình đã train
model = tf.keras.models.load_model('video_recommendation_model.h5')

# Hàm ánh xạ tên thể loại sang index
def genre_to_index(genre):
    genre_map = {'action': 0, 'anime': 1}  # Cập nhật với ánh xạ thực tế
    return genre_map.get(genre, -1)

# Hàm tạo embedding vector từ input
def create_input_vector(genres_watched):
    input_vector = np.zeros((1, num_genres))
    for genre, count in genres_watched.items():
        genre_idx = genre_to_index(genre)
        if genre_idx != -1:  # Nếu thể loại tồn tại
            input_vector[0, genre_idx] = count
    
    # Embedding step: Giả sử embedding layer là một phần của mô hình
    # Bạn có thể sử dụng mô hình hoặc thêm bước xử lý tại đây nếu cần
    # return np.dot(input_vector, embedding_matrix)  # Nếu cần

    return input_vector[:, :embedding_size]  # Giả sử bạn cần giảm xuống 128

# API nhận số lần xem các thể loại và trả về các thể loại được đề xuất
@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.json
    genres_watched = data['genres_watched']  # Nhận dữ liệu: {"action": 3, "anime": 2}
    
    # Tạo input vector từ số lần xem các thể loại
    input_vector = create_input_vector(genres_watched)
    
    # Dự đoán từ mô hình AI
    predicted_genres = model.predict(input_vector)
    
    # Lấy top 5 thể loại được đề xuất
    recommended_genres = np.argsort(predicted_genres[0])[-5:]
    
    return jsonify({'recommended_genres': recommended_genres.tolist()})

if __name__ == '__main__':
    app.run(debug=True, port=5050)
