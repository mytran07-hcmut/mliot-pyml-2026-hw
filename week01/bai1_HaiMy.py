import numpy as np

#Convert danh sach (cau) thanh 1 ma tran so (X)
cau = [
    "I love you",
    "Ho Chi Minh University of Technology is a reputative university",
    "I'm majoring in AI",
    "I will graduate in 2028",
    "I'm from Ho Chi Minh city, VietNam",
    "I'm interested in Machine Learning and AI",
    "My hobby is playing sport",
    "I can play many musical instruments"
]
vocab = sorted({w for s in cau for w in s.lower().split()}) #Sap xep cac tu duoc tach ra tu cac cau va xep vao mot set
#Convert 1 cau thanh 1 vector
def to_vector(s):
    v = np.zeros(len(vocab))
    for w in s.lower().split():
        v[vocab.index(w)] += 1
    return v
X = np.array([to_vector(s) for s in cau])

#Cau 1: Tạo ma trận X: với text là (số câu × số từ).
#In X.shape và giải thích mỗi hàng/cột đại diện cho gì.
print("CAU 1:")
print(X.shape)
print("Mỗi hàng đại diện cho 1 câu (8 câu)")
print("Mỗi cột đại diện cho 1 từ (35 từ)")

#Cau 2: Tính vector trung bình theo cột,
#trừ trung bình (broadcasting); in shape trước/sau để minh họa quy tắc broadcasting.
print("CAU 2:")
X_mean = np.mean(X, axis=0)
print("Shape của vector trung bình X_mean:", X_mean.shape)
X_centered = X - X_mean
print("Shape của X sau khi trừ trung bình X_centered:", X_centered.shape)

#Cau 3: Viết hàm cosine_similarity(X, Y=None)
#cho batch (chuẩn hóa theo hàng, keepdims=True), trả về ma trận tương đồng.
print("CAU 3:")
def cosine_similarity(X, Y=None):
    if Y is None:
        Y = X
    #Chuẩn hoá ma trận X theo từng hàng
    Xn = X / np.linalg.norm(X, axis=1, keepdims=True)
    Yn = Y / np.linalg.norm(Y, axis=1, keepdims=True)
    return Xn @ Yn.T #Ma trận tương đồng
S = cosine_similarity(X)

#Cau 4: Viết hàm search(query, top_k=3) trả về top_k mục
#giống truy vấn nhất kèm điểm cosine
print("CAU 4:")
def search(query, top_k=3):
    #Convert câu query input vào thành vector
    q_vector = to_vector(query).reshape(1,-1)

    #Tính chỉ số tương đồng giữa query và ma trận X
    similarity = cosine_similarity(X,q_vector).flatten()

    #Lấy ra top_k chỉ số có điểm cao nhất (sắp xếp giảm dần)
    top_indices = np.argsort(similarity)[::-1][:top_k]

    #Print kết quả
    print("Kết quả tìm kiếm cho query: ", query)
    for idx in top_indices:
        print(f"- Câu: '{cau[idx]}' | Điểm Cosine: {similarity[idx]:.4f}")

search("I love Machine Learning and AI", top_k=3)

#Cau 5: Tìm cặp câu giống nhau nhất và cặp câu khác nhau nhất
#rồi nhận xét kết quả trả về so với thực tế
print("CAU 5:")
#Tìm cặp câu giống nhau nhất (argmax())
S_max = S.copy()
np.fill_diagonal(S_max, -1) #Đặt đường chéo chính bằng -1 để không tính trùng chính nó
idx_max = np.unravel_index(np.argmax(S_max), S_max.shape)
print(f"Cặp câu giống nhau nhất (Điểm: {S_max[idx_max]:.4f}): ")
print(f" 1. '{cau[idx_max[0]]}'")
print(f" 2. '{cau[idx_max[1]]}'")

#Tìm cặp câu khác nhau nhất (argmin())
S_min = S.copy()
np.fill_diagonal(S_min, 100) #Đặt đường chéo chính bằng 100 để không tính trùng chính nó
idx_min = np.unravel_index(np.argmin(S_min), S_min.shape)
print(f"Cặp câu khác nhau nhất (Điểm: {S_min[idx_min]:.4f}): ")
print(f" 1. '{cau[idx_min[0]]}'")
print(f" 2. '{cau[idx_min[1]]}'")

#Nhận xét:
# - Cặp giống nhau nhất: Là cặp câu có chung nhiều từ khoá nhất ("Machine Learning" và "AI"), khớp với trực giác
#thực tế vì chúng là những từ vựng trùng lặp nên chắc chắn có cùng ngữ cảnh
# - Cặp khác nhau nhất: Là cặp câu không có bất kỳ một từ nào trùng nhau hết nhưng trong thực tế thì có
#thể có trường hợp tuy các từ không trùng nhau nhưng có cùng ý nghĩa về ngữ cảnh (ví dụ: "piano" và "guitar")
#nên có thể kiểu tìm này sẽ không đảm bảo được tính chính xác so với thực tế
