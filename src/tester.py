# Using my_model.keras for the specified index from data/train.csv
import pandas as pd
from tensorflow import keras
from sklearn.preprocessing import StandardScaler

model = keras.models.load_model('model/my_model.keras')
data = pd.read_csv('data/train.csv')

index = 0  # Specify the index of the row you want to test (index + 2 is the line number in the CSV)
row = data.iloc[index, 0:20].values.astype(float).reshape(1, -1)  # First 20 columns as features
true_label = data.iloc[index, 20]  # 21st column as true label

# Standardize using StandardScaler fitted on all training data
scaler = StandardScaler()
scaler.fit(data.iloc[:, 0:20].values.astype(float))
row_scaled = scaler.transform(row)

predicted_prob = model.predict(row_scaled)[0][0]
predicted_label = 'male' if predicted_prob >= 0.5 else 'female'

# Print row
print(f"Predicted Probability: {predicted_prob:.4f}")
print(f"Predicted Label: {predicted_label}")
print(f"True Label: {true_label}")