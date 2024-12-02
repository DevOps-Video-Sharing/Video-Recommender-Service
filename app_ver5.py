from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
import json
from sklearn.metrics.pairwise import cosine_similarity
import pickle
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

@app.route('/synonyms', methods=['POST'])
def get_synonyms():
    data = request.json
    words = data.get('words', [])  # Lấy danh sách từ từ payload
    
    if not isinstance(words, list) or not words:
        return jsonify({'error': 'Input must be a non-empty array of words'}), 400
    
    synonyms_list = []
    
    for word in words:
        word = word.strip()
        if word:  # Bỏ qua các từ rỗng
            synonyms = find_synonyms(word)
            synonyms_list.extend(synonyms)  # Gộp tất cả các từ đồng nghĩa vào danh sách
    
    return jsonify(synonyms_list)





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