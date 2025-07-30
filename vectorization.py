from sentence_transformers import SentenceTransformer, util

# 사전 훈련된 임베딩 모델 로드
en_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
ko_model = SentenceTransformer('jhgan/ko-sroberta-multitask')

# 문장 임베딩
sentence_1_en = "dog is cute"
sentence_2_en = "dog is lovely"
sentence_3_en = "car is fast"

sentence_1_ko = "강아지는 귀엽다"
sentence_2_ko = "강아지는 사랑스럽다"
sentence_3_ko = "자동차는 빠르다"

en_embeddings = en_model.encode([sentence_1_en, sentence_2_en, sentence_3_en], convert_to_tensor=True)
ko_embeddings = ko_model.encode([sentence_1_ko, sentence_2_ko, sentence_3_ko], convert_to_tensor=True)

# 영어 유사도 계산 (코사인 유사도)
cos_sim_1_2 = util.cos_sim(en_embeddings[0], en_embeddings[1])
cos_sim_1_3 = util.cos_sim(en_embeddings[0], en_embeddings[2])

# 한국어 유사도 계산 (코사인 유사도)
cos_sim_1_2_ko = util.cos_sim(ko_embeddings[0], ko_embeddings[1])
cos_sim_1_3_ko = util.cos_sim(ko_embeddings[0], ko_embeddings[2])

# 영어 문장 유사도 평가
print(f"'{sentence_1_en}' vs '{sentence_2_en}' → 유사도: {cos_sim_1_2.item():.4f}")
print(f"'{sentence_1_en}' vs '{sentence_3_en}' → 유사도: {cos_sim_1_3.item():.4f}")
# 한국어 문장 유사도 평가
print(f"'{sentence_1_ko}' vs '{sentence_2_ko}' → 유사도: {cos_sim_1_2_ko.item():.4f}")
print(f"'{sentence_1_ko}' vs '{sentence_3_ko}' → 유사도: {cos_sim_1_3_ko.item():.4f}")
