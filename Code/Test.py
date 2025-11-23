import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.random_projection import GaussianRandomProjection
from sklearn.metrics import r2_score

# 1. ĐỌC DỮ LIỆU
df = pd.read_csv('LaptopPrice.csv')
X = df.drop('Price', axis=1) # 6 cột đặc trưng (RAM, CPU...)
y = df['Price']              # Cột giá tiền

# Chia tập huấn luyện và kiểm tra (80% học, 20% thi)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ---------------------------------------------------------
# TRƯỜNG HỢP 1: DỮ LIỆU GỐC (6 CHIỀU)
# ---------------------------------------------------------
model_orig = LinearRegression()
model_orig.fit(X_train, y_train)
y_pred_orig = model_orig.predict(X_test)
score_orig = r2_score(y_test, y_pred_orig)

# ---------------------------------------------------------
# TRƯỜNG HỢP 2: DÙNG RANDOM PROJECTION (NÉN CÒN 3 CHIỀU)
# ---------------------------------------------------------
# Khởi tạo bộ chiếu ngẫu nhiên Gaussian để giảm từ 6 -> 3 chiều
rp = GaussianRandomProjection(n_components=3, random_state=42)

# Thực hiện chiếu (nén) dữ liệu
X_train_rp = rp.fit_transform(X_train)
X_test_rp = rp.transform(X_test)

# Huấn luyện lại trên dữ liệu nén
model_rp = LinearRegression()
model_rp.fit(X_train_rp, y_train)
y_pred_rp = model_rp.predict(X_test_rp)
score_rp = r2_score(y_test, y_pred_rp)

# ---------------------------------------------------------
# XUẤT KẾT QUẢ
# ---------------------------------------------------------
print("-" * 40)
print(f"Độ chính xác (R2) gốc (6 chiều): {score_orig:.4f}")
print(f"Độ chính xác (R2) nén (3 chiều): {score_rp:.4f}")
print("-" * 40)
print("Nhận xét: Dữ liệu giảm đi 50% dung lượng, độ chính xác có giảm nhưng vẫn chấp nhận được cho mục đích lưu trữ/xử lý nhanh.")