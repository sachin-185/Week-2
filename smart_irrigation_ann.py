import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("data.csv")

# Drop unwanted column
df = df.drop(columns=["Unnamed: 0"])

# Features: sensor columns | Targets: parcel columns
X = df[[col for col in df.columns if 'sensor' in col]]
y = df[[col for col in df.columns if 'parcel' in col]]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ANN Model: MLPRegressor
model = MLPRegressor(hidden_layer_sizes=(64, 32), activation='relu', max_iter=500, random_state=42)
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Histogram Comparison: Actual vs Predicted for Each Parcel
for i, col in enumerate(y.columns):
    plt.figure(figsize=(8, 4))
    plt.hist(y_test[col], bins=20, alpha=0.6, label="Actual", color='skyblue', edgecolor='black')
    plt.hist(y_pred[:, i], bins=20, alpha=0.6, label="Predicted", color='salmon', edgecolor='black')
    plt.title(f"Histogram: Actual vs Predicted ({col})")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()
    plt.show()
