# Import
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# FIX SEED (ổn định kết quả)
import random
import tensorflow as tf
seed = 42
np.random.seed(seed)
tf.random.set_seed(seed)
random.seed(seed)
# Import
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# FIX SEED (ổn định kết quả)
import random
import tensorflow as tf
seed = 42
np.random.seed(seed)
tf.random.set_seed(seed)
random.seed(seed)
df = yf.download("VNM.VN", start="2018-01-01", end="2026-10-14")
data = df[['Close']].values
train_size = int(0.8 * len(data))
train_data = data[:train_size]
test_data = data[train_size:]
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
train_scaled = scaler.fit_transform(train_data)
test_scaled = scaler.transform(test_data)
def create_sequences(data, window_size=60):
    X, y = [], []
    for i in range(window_size, len(data)):
        X.append(data[i-window_size:i])
        y.append(data[i])
    return np.array(X), np.array(y)
X_train, y_train = create_sequences(train_scaled)
test_combined = np.vstack((train_scaled[-60:], test_scaled))
X_test, y_test = create_sequences(test_combined)
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
val_size = int(0.1 * len(X_train))
X_val = X_train[-val_size:]
y_val = y_train[-val_size:]
X_train_final = X_train[:-val_size]
y_train_final = y_train[:-val_size]
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(60, 1)))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')
from tensorflow.keras.callbacks import EarlyStopping
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)
history = model.fit(
    X_train_final, y_train_final,
    validation_data=(X_val, y_val),
    epochs=10,
    batch_size=32,
    shuffle=False,   
    callbacks=[early_stop]
)
test_pred = model.predict(X_test)
test_pred = scaler.inverse_transform(test_pred)
y_test_real = scaler.inverse_transform(y_test.reshape(-1,1))
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
rmse = np.sqrt(mean_squared_error(y_test_real, test_pred))
mae = mean_absolute_error(y_test_real, test_pred)
r2 = r2_score(y_test_real, test_pred)
print("RMSE:", rmse)
print("MAE:", mae)
print("R2:", r2)
plt.figure(figsize=(10,5))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title("Loss Train vs Validation")
plt.show()
plt.figure(figsize=(12,6))
plt.plot(y_test_real, label="Giá thật")
plt.plot(test_pred, label="Dự báo")
plt.legend()
plt.show()
last_60_days = np.vstack((train_scaled, test_scaled))[-60:]
last_60_days = last_60_days.reshape(1, 60, 1)
future_predictions = []
for i in range(10):
    next_pred = model.predict(last_60_days)
    future_predictions.append(next_pred[0,0])
    last_60_days = np.append(
        last_60_days[:,1:,:],
        next_pred.reshape(1,1,1),
        axis=1
    )
future_predictions = np.array(future_predictions).reshape(-1,1)
future_predictions = scaler.inverse_transform(future_predictions)
print("Dự báo 10 ngày tới:")
print(future_predictions)
plt.figure(figsize=(12,6))
plt.plot(y_test_real, label="Giá thật")
plt.plot(test_pred, label="Dự báo test")
future_x = range(len(y_test_real), len(y_test_real)+10)
plt.plot(future_x, future_predictions, marker='o', label="10 ngày tới")
plt.legend()
plt.show()
