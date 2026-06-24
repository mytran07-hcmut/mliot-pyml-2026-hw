import numpy as np
import matplotlib.pyplot as plt

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

#Cau 2: Đưa mỗi câu về 2 chiều (LSA thu nhỏ)
Xc = X - X.mean(axis=0)
U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
# Cách chuẩn LSA: Nhân 2 cột đầu của U với ma trận vuông chứa 2 giá trị trị riêng lớn nhất của Sigma
coords = U[:, :2] * S[:2] #Toạ độ 2 chiều của mỗi câu

#Cau 3: Vẽ scatter các câu trên mặt phẳng 2D, gắn nhãn từng câu
plt.figure(figsize=(10, 8)) #

# Vẽ scatter các câu
plt.scatter(coords[:, 0], coords[:, 1], color='red', edgecolors='k', s=100, zorder=3)

# Gắn nhãn từng câu
for i, txt in enumerate(cau):
    # Để tránh chữ bị đè lên dấu chấm, ta dịch nhẹ tọa độ text qua phải một chút (+0.05)
    plt.annotate(txt, (coords[i, 0], coords[i, 1]), fontsize=10, weight='bold')

plt.title("Visualization", fontsize=14, pad=15)
plt.xlabel("Component 1", fontsize=12)
plt.ylabel("Component 2", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.axhline(0, color='black',linewidth=0.5)
plt.axvline(0, color='black',linewidth=0.5)

# Hiển thị biểu đồ
plt.show()

#Câu 4: nhận xét
# - Các câu có chung từ khoá (chủ đề) với nhau thì các points
#của các câu đó có xu hướng nằm gần nhau (co cụm)
# - Liên hệ với PCA/eigen: Việc lấy 2 cột đầu của ma trận U nhân ma trận sigma cũng giống với
# phép chiếu dữ liệu lên các trục chính trong PCA - nơi mà dữ liệu có variance lớn nhất
# và các eigenvalues nằm trong ma trận sigma thể hiện lượng thông tin mà mỗi component đó nắm giữ