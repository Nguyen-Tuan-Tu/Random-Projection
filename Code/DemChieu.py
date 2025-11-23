import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.random_projection import GaussianRandomProjection
from sklearn.metrics.pairwise import euclidean_distances

# 1. ĐỌC FILE CSV
# Thay đúng tên file csv của bạn vào đây
df = pd.read_csv('test.csv')

# --- SỬA LỖI Ở ĐÂY ---
# Thay vì cắt thủ công, ta dùng lệnh này để chỉ lấy các cột là SỐ (float, int)
# Nó sẽ tự động bỏ qua các cột chữ như 'STANDING', 'WALKING'...
data_numeric = df.select_dtypes(include=[np.number])

print(f"Kích thước dữ liệu sau khi lọc bỏ cột chữ: {data_numeric.shape}")
# Kiểm tra xem có bị mất hết dữ liệu không (đề phòng trường hợp file toàn chữ)
if data_numeric.shape[1] == 0:
    print("LỖI: Không tìm thấy cột số nào! Hãy kiểm tra lại file CSV.")
    exit()

# Lấy 500 mẫu đầu tiên để vẽ
sample_data = data_numeric.iloc[:500]

# 2. TÍNH KHOẢNG CÁCH GỐC
print("Đang tính khoảng cách gốc...")
dist_original = euclidean_distances(sample_data)
flat_dist_original = dist_original[np.triu_indices(500, k=1)]

# 3. ÁP DỤNG RANDOM PROJECTION (Nén xuống 50 chiều)
print("Đang thực hiện Random Projection...")
rp = GaussianRandomProjection(n_components=50, random_state=42)
data_projected = rp.fit_transform(sample_data)

# 4. TÍNH KHOẢNG CÁCH MỚI
dist_projected = euclidean_distances(data_projected)
flat_dist_projected = dist_projected[np.triu_indices(500, k=1)]

# 5. VẼ BIỂU ĐỒ
print("Đang vẽ biểu đồ...")
plt.figure(figsize=(8, 8))
plt.scatter(flat_dist_original, flat_dist_projected, alpha=0.1, s=1, color='blue')
plt.title(f"Kiểm chứng Bổ đề JL (Nén {data_numeric.shape[1]} chiều -> 50 chiều)")
plt.xlabel("Khoảng cách Gốc")
plt.ylabel("Khoảng cách Sau nén")

# Vẽ đường chéo đỏ minh họa xu hướng
try:
    m, b = np.polyfit(flat_dist_original, flat_dist_projected, 1)
    plt.plot(flat_dist_original, m*flat_dist_original + b, color='red', linestyle='--')
except:
    pass

plt.grid(True)
plt.tight_layout()
plt.show()

# Tính độ tương quan
correlation = np.corrcoef(flat_dist_original, flat_dist_projected)[0, 1]
print("-" * 30)
print(f"Độ tương quan (Correlation): {correlation:.4f}")