import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 1. CỐ ĐỊNH SEED (Đảm bảo tính ổn định kết quả)
seed = 42
np.random.seed(seed)
tf.random.set_seed(seed)
random.seed(seed)

# 2. NHẬP MÃ CỔ PHIẾU VÀ TẢI DỮ LIỆU THỰC TẾ
symbol = input("Nhập mã cổ phiếu (ví dụ: VNM.VN, HPG.VN, FPT.VN, AAPL): ").strip().upper()
start_date = "2018-01-01"
end_date = "2026-10-14"

print(f"\n🔄 Đang tải dữ liệu thực tế cho mã [{symbol}] từ Yahoo Finance...")
df = yf.download(symbol, start=start_date, end=end_date)

# Kiểm tra xem mã cổ phiếu có tồn tại và tải được dữ liệu không
if df.empty:
    raise ValueError(f"❌ Không tìm thấy dữ liệu cho mã '{symbol}'. Vui lòng kiểm tra lại mã (Lưu ý: Cổ phiếu Việt Nam cần thêm đuôi .VN, ví dụ: VNM.VN).")

print(f"✅ Đã tải thành công {len(df)} dòng dữ liệu từ {start_date} đến {end_date}.")

# Trích xuất giá đóng cửa (Close)
data = df[['Close']].values

# 3. PHÂN CHIA VÀ CHUẨN HÓA DỮ LIỆU
train_size = int(0.8 * len(data))
train_data = data[:train_size]
test_data = data[train_size:]

scaler = MinMaxScaler()
train_scaled = scaler.fit_transform(train_data)
test_scaled = scaler.transform(test_data)

# Hàm tạo cửa sổ trượt (Sliding Window)
def create_sequences(data, window_size=60):
    X, y = [], []
    for i in range(window_size, len(data)):
        X.append(data[i-window_size:i])
        y.append(data[i])
    return np.array(X), np.array(y)

X_train, y_train = create_sequences(train_scaled)
test_combined = np.vstack((train_scaled[-60:], test_scaled))
X_test, y_test = create_sequences(test_combined)

# Reshape dữ liệu về dạng 3D [Samples, Timesteps, Features] cho LSTM
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# Tách tập Validation
val_size = int(0.1 * len(X_train))
X_val = X_train[-val_size:]
y_val = y_train[-val_size:]
X_train_final = X_train[:-val_size]
y_train_final = y_train[:-val_size]

# 4. XÂY DỰNG VÀ HUẤN LUYỆN MÔ HÌNH LSTM
print("\n🧠 Đang huấn luyện mô hình LSTM...")
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(60, 1)),
    LSTM(50),
    Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

history = model.fit(
    X_train_final, y_train_final,
    validation_data=(X_val, y_val),
    epochs=15,
    batch_size=32,
    shuffle=False,
    callbacks=[early_stop],
    verbose=1
)

# 5. ĐÁNH GIÁ MÔ HÌNH TRÊN TẬP TEST
test_pred = model.predict(X_test)
test_pred = scaler.inverse_transform(test_pred)
y_test_real = scaler.inverse_transform(y_test.reshape(-1, 1))

rmse = np.sqrt(mean_squared_error(y_test_real, test_pred))
mae = mean_absolute_error(y_test_real, test_pred)
r2 = r2_score(y_test_real, test_pred)

print(f"\n📊 BÁO CÁO ĐÁNH GIÁ MÔ HÌNH [MÃ: {symbol}]")
print(f"• RMSE: {rmse:.2f}")
print(f"• MAE:  {mae:.2f}")
print(f"• R2:   {r2:.4f}")

# 6. DỰ BÁO 10 NGÀY TƯƠNG LAI
last_60_days = np.vstack((train_scaled, test_scaled))[-60:]
last_60_days = last_60_days.reshape(1, 60, 1)

future_predictions = []
for _ in range(10):
    next_pred = model.predict(last_60_days, verbose=0)
    future_predictions.append(next_pred[0, 0])
    last_60_days = np.append(last_60_days[:, 1:, :], next_pred.reshape(1, 1, 1), axis=1)

future_predictions = np.array(future_predictions).reshape(-1, 1)
future_predictions = scaler.inverse_transform(future_predictions)

print(f"\n🔮 BẢNG DỰ BÁO GIÁ GIÁ 10 NGÀY TỚI CHO [{symbol}]:")
for idx, price in enumerate(future_predictions, 1):
    print(f"  • Ngày +{idx}: {price[0]:,.2f}")

# 7. TRỰC QUAN HÓA KẾT QUẢ
plt.figure(figsize=(10, 4))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title(f"Hàm Mất Mát (Loss) Của Mô Hình - {symbol}")
plt.legend()
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(y_test_real, label="Giá thực tế", color='blue')
plt.plot(test_pred, label="Giá dự báo (Test)", color='orange')
future_x = range(len(y_test_real), len(y_test_real) + 10)
plt.plot(future_x, future_predictions, marker='o', label="Dự báo 10 ngày tới", color='green')
plt.title(f"Dự Báo Giá Cổ Phiếu [{symbol}] Bằng LSTM")
plt.xlabel("Thời gian (ngày)")
plt.ylabel("Giá đóng cửa")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()
