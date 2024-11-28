from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np

# Load mô hình đã train
model = tf.keras.models.load_model('video_recommendation_model_ver2.h5')

app = Flask(__name__)

# Định nghĩa route cho API
@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        # Nhận dữ liệu từ request
        data = request.json
        user_history = data['user_history']  # Lịch sử xem video của người dùng

        # Tiền xử lý dữ liệu trước khi đưa vào mô hình
        input_data = np.array(user_history).reshape(1, -1)  # Reshape để phù hợp với input của mô hình
        
        # Dự đoán video đề xuất
        predictions = model.predict(input_data)
        
        # Chuyển kết quả về danh sách video đề xuất
        recommended_videos = predictions[0].argsort()[-10:][::-1]  # Lấy top 10 video

        return jsonify({'recommended_videos': recommended_videos.tolist()})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5050)
