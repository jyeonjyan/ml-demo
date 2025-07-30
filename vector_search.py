from sentence_transformers import SentenceTransformer
import faiss

# 여기에 직접 문장들을 넣으세요
sentences = [
    "강아지는 귀엽다",
    "개는 사랑스럽다",
    "고양이는 도도하다",
    "자동차는 빠르다",
    "산은 높고 푸르다"
]

# 한국어 지원 임베딩 모델
model = SentenceTransformer('jhgan/ko-sroberta-multitask')

# 문장 임베딩
embeddings = model.encode(sentences, convert_to_numpy=True).astype('float32')

# 코사인 유사도 계산을 위한 벡터 정규화
faiss.normalize_L2(embeddings)

# FAISS(파이스) 인덱스 생성
dimension = embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)  # IP = Inner Product = Cosine 유사도용
index.add(embeddings)

# 쿼리: 직접 검색할 문장을 여기에 입력
query = "멋있는 자동차"
query_vec = model.encode([query], convert_to_numpy=True).astype('float32')
faiss.normalize_L2(query_vec)

# 검색 실행
top_k = 3
distances, indices = index.search(query_vec, top_k)

# 결과 출력
print(f"쿼리: '{query}'")
for i in range(top_k):
    idx = indices[0][i]
    distance = distances[0][i]
    print(f"유사도: {distance:.4f}, 문장: '{sentences[idx]}' (인덱스 {idx} 에 위치)")
