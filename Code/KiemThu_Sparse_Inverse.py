import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.random_projection import SparseRandomProjection
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics import mean_squared_error

# 1. ĐỌC DỮ LIỆU
# Nhớ thay tên file của bạn vào đây
df = pd.read_csv('test.csv') # Ví dụ tên file là train.csv

# Lấy dữ liệu số (bỏ cột chữ)
X = df.select_dtypes(include=[np.number])
# Bỏ cột 'subject' nếu có (vì nó là mã định danh, không phải tín hiệu)
if 'subject' in X.columns:
    X = X.drop('subject', axis=1)

print(f"Dữ liệu gốc: {X.shape} (561 chiều)")
sample_data = X.iloc[:500] # Lấy 500 mẫu để test

# ==============================================================================
# PHẦN 1: SPARSE RANDOM PROJECTION
# ==============================================================================
print("\n--- Đang chạy Sparse Random Projection ---")
# 'density=auto': Tự động tính độ thưa tối ưu theo công thức JL
srp = SparseRandomProjection(n_components=50, density='auto', random_state=42)

# Nén dữ liệu (Transform)
X_projected = srp.fit_transform(sample_data)
print(f"Dữ liệu sau khi nén: {X_projected.shape} (50 chiều)")

# Kiểm tra chất lượng nén (Correlation)
dist_orig = euclidean_distances(sample_data)
dist_proj = euclidean_distances(X_projected)

flat_orig = dist_orig[np.triu_indices(500, k=1)]
flat_proj = dist_proj[np.triu_indices(500, k=1)]
corr = np.corrcoef(flat_orig, flat_proj)[0, 1]
print(f"Hệ số tương quan (Correlation) của Sparse RP: {corr:.4f}")
print("(Nếu > 0.85 là Sparse hoạt động rất tốt!)")

# ==============================================================================
# PHẦN 2: INVERSE TRANSFORM (KHÔI PHỤC DỮ LIỆU)
# ==============================================================================
print("\n--- Đang chạy Inverse Transform ---")

# Khôi phục lại dữ liệu từ 50 chiều -> 561 chiều
X_reconstructed = srp.inverse_transform(X_projected)
print(f"Kích thước dữ liệu sau khi khôi phục: {X_reconstructed.shape}")

# Tính sai số tái tạo (Reconstruction Error)
mse = mean_squared_error(sample_data, X_reconstructed)
print(f"Sai số trung bình bình phương (MSE): {mse:.4f}")

# ==============================================================================
# VẼ BIỂU ĐỒ SO SÁNH (VISUALIZATION)
# ==============================================================================
# Lấy ngẫu nhiên 1 mẫu (ví dụ mẫu đầu tiên) để vẽ đường tín hiệu
idx = 0
original_signal = sample_data.iloc[idx].values
reconstructed_signal = X_reconstructed[idx]

plt.figure(figsize=(12, 6))
plt.plot(original_signal, label='Dữ liệu Gốc (Original)', color='blue', alpha=0.7)
plt.plot(reconstructed_signal, label='Dữ liệu Khôi phục (Reconstructed)', color='red', alpha=0.7, linestyle='--')
plt.title(f"So sánh Tín hiệu Gốc vs Khôi phục (Nén 561 -> 50 -> 561)\nMẫu số {idx}")
plt.xlabel("Chỉ số đặc trưng (Features Index)")
plt.ylabel("Giá trị")
plt.legend()
plt.grid(True)
plt.show()