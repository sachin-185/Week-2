import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor

df = pd.read_csv("data.csv")
if "Unnamed: 0" in df.columns:
    df = df.drop(columns=["Unnamed: 0"])

sensor_cols = [c for c in df.columns if "sensor" in c]
parcel_cols = [c for c in df.columns if "parcel" in c]

X = df[sensor_cols]
y = df[parcel_cols]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = MLPRegressor(hidden_layer_sizes=(64, 32), activation='relu', max_iter=500, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

for i, col in enumerate(parcel_cols):
    plt.figure(figsize=(8, 4))
    plt.hist(y_test[col], bins=20, alpha=0.6, label="Actual", color='skyblue', edgecolor='black')
    plt.hist(y_pred[:, i], bins=20, alpha=0.6, label="Predicted", color='salmon', edgecolor='black')
    plt.title(f"Actual vs Predicted ({col})")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()
    plt.show()