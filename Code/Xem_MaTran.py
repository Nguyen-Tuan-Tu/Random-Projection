import pandas as pd
import numpy as np

# 1. Đọc dữ liệu
df = pd.read_csv('test.csv')

# 2. Lấy phần lõi số (Bỏ cột chữ 'STANDING' đi)
# Đây chính là bước chuyển từ bảng Excel sang Ma trận Toán học
X = df.select_dtypes(include=[np.number]).values

# 3. In thông tin ma trận
print("--- CẤU TRÚC MA TRẬN ---")
print(f"Kích thước ma trận (Dòng x Cột): {X.shape}")
print(f"Kiểu dữ liệu trong ma trận: {X.dtype}")
print("\n--- HÌNH DẠNG THỰC TẾ (5 dòng đầu, 5 cột đầu) ---")
# In thử một góc nhỏ của ma trận để xem
print(X[:5, :5])

print("\n--- GIẢI THÍCH ---")
print(f"Mỗi dòng (ví dụ dòng 0) là một vector trong không gian {X.shape[1]} chiều:")
print(X[0])