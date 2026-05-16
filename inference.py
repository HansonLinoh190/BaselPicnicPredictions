import joblib
import time
import numpy as np

model = joblib.load("ML/Session 10/picnic_model.pkl")

# Data
test_data = np.array([[1, 8, 0.89, 10.286, 0.2, 0.03, 0.0, 25, 70, 3.9]]) 

# Start time
start_time = time.time()

# Predict
prediction = model.predict(test_data)

# Finish time
end_time = time.time()

# Latency
latency = (end_time - start_time) * 1000

print("Prediksi cuaca:", prediction[0])
print(f"Latency: {latency:.4f} ms")
