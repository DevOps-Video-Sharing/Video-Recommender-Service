from kafka import KafkaConsumer, KafkaProducer
from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
import json
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import threading
app = Flask(__name__)

# Load model
model = tf.keras.models.load_model('synonym_model.h5')

# Load từ điển word_to_index và index_to_word
with open('word_to_index.pkl', 'rb') as f:
    word_to_index = pickle.load(f)

with open('index_to_word.pkl', 'rb') as f:
    index_to_word = pickle.load(f)

# Lấy ma trận nhúng từ lớp Embedding
embedding_matrix = model.get_layer('embedding_2').get_weights()[0]

# Hàm tìm từ đồng nghĩa
def find_synonyms(word, top_n=5):
    if word not in word_to_index:
        return []
    
    word_idx = word_to_index[word]
    word_vector = embedding_matrix[word_idx].reshape(1, -1)
    similarities = cosine_similarity(word_vector, embedding_matrix)[0]
    similar_indices = np.argsort(-similarities)[1:top_n + 1]
    similar_words = [index_to_word[idx] for idx in similar_indices]
    
    return similar_words

# Kafka Consumer
def consume_kafka():
    consumer = KafkaConsumer(
        'synonyms_topic',
        bootstrap_servers=['localhost:9092'],
        auto_offset_reset='earliest',
        value_deserializer=lambda x: x.decode('utf-8')
    )

    print("Listening to Kafka topic: synonyms_topic")
    for message in consumer:
        raw_value = message.value
        print(f"Raw Value: {raw_value}")

        # Parse JSON từ Kafka
        parsed_data = convert_to_json(raw_value)
        if not parsed_data:
            print("Invalid JSON format. Skipping message.")
            continue

        # Xử lý tin nhắn
        processed_data = process_message(parsed_data)
        print(f"Processed Data: {json.dumps(processed_data, indent=4)}")
        user_id = processed_data.get("userId", "")
        video_data = processed_data.get("videoData", [])
        produce_message(user_id, video_data)


def process_message(parsed_data):
    user_id = parsed_data.get("userId", "").strip()
    video_data = parsed_data.get("videoData", {})

    if not isinstance(video_data, dict):
        print("Invalid videoData format. Skipping message.")
        return {"error": "Invalid videoData format"}

    # Loại bỏ giá trị null hoặc None
    video_data_cleaned = {
        key: value for key, value in video_data.items() if value is not None
    }

    # Tạo danh sách từ đồng nghĩa tổng hợp
    combined_synonyms = set()  # Sử dụng set để tránh trùng lặp
    for key in video_data_cleaned.keys():
        words = key.split()
        for word in words:
            word = word.strip()
            if word:  # Bỏ qua từ rỗng
                synonyms = find_synonyms(word)
                combined_synonyms.add(word)  # Thêm từ chính
                combined_synonyms.update(synonyms)  # Thêm từ đồng nghĩa

    # Trả về kết quả đã xử lý
    result = {
        "userId": user_id,
        "videoData": list(combined_synonyms)  # Chuyển set thành danh sách
    }
    return result



def convert_to_json(raw_value):
    try:
        return json.loads(raw_value)
    except json.JSONDecodeError as e:
        print(f"Error converting to JSON: {e}")
        return None


@app.route('/process_synonyms', methods=['POST'])
def process_synonyms():
    data = request.json  # Lấy dữ liệu đầu vào

    if not isinstance(data, dict) or not data:
        return jsonify({'error': 'Input must be a non-empty JSON object'}), 400

    # Xử lý userId
    user_id = data.get("userId", "").strip()

    # Xử lý videoData
    video_data = data.get("videoData", {})
    if not isinstance(video_data, dict):
        return jsonify({'error': '"videoData" must be a JSON object'}), 400

    # Tìm từ đồng nghĩa cho các khóa của videoData
    synonyms_result = {}
    for key in video_data.keys():
        words = key.split()  # Tách các từ trong khóa
        for word in words:
            word = word.strip()  # Loại bỏ khoảng trắng
            if word:  # Bỏ qua từ rỗng
                synonyms = find_synonyms(word)
                if word not in synonyms_result:  # Tránh trùng lặp
                    synonyms_result[word] = synonyms

    # Kết quả đầu ra
    result = {
        "userId": user_id,
        "videoData": synonyms_result
    }
    
    return jsonify(result)



# Chạy Kafka Consumer trong một thread riêng
kafka_thread = threading.Thread(target=consume_kafka, daemon=True)
kafka_thread.start()

def produce_message(user_id, genres):
    producer = KafkaProducer(
        bootstrap_servers=['localhost:9092'],
        value_serializer=lambda v: json.dumps(v).encode('utf-8')
    )

    message = {
        "userId": user_id,
        "genres": genres
    }
    producer.send('recommendation_topic', value=message)
    producer.flush()
    print(f"Message sent: {message}")


#####################################################################################


from keybert import KeyBERT
kw_model = KeyBERT()

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

    # kafka_threadSynonyms = threading.Thread(target=consume_kafkaSynonyms)
    # kafka_threadSynonyms.start()
    app.run(host='0.0.0.0', port=5050)