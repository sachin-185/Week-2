# 🌱 Smart Irrigation System (Week-2)

Now, this project implements a **Smart Irrigation System** using **Predictive Modeling** with an **Artificial Neural Network (ANN)** to automate water supply decisions based on real-time environmental data. By leveraging machine learning it helps farmers optimize irrigation schedules, reduce water waste, and support sustainable agriculture.

---

## 📊 Model Training Summary

The ANN model was trained to predict whether irrigation is needed (`1`) or not (`0`) using environmental sensor data. Key training highlights:

- 🔁 **Training epochs**: 20  
- 🎯 **Initial accuracy**: 64%  
- ✅ **Final training accuracy**: 94%  
- 🧪 **Test accuracy**: 87%

These results indicate the model’s strong ability to **learn from data** and **generalize well** to new, unseen conditions.

---

## 🔍 How It Works

This predictive model uses environmental sensor inputs such as:

- 🌡️ Temperature  
- 💧 Soil Moisture  
- 💦 Humidity  

The trained ANN processes this data and outputs:

- `1` → **Irrigation needed**  
- `0` → **No irrigation needed**

This enables:

- ✅ **Efficient water use**
- ✅ **Reduced manual intervention**
- ✅ **Informed, data-driven farming**

---

## ✅ Key Features

- 🤖 **Predictive binary classification** using ANN  
- 🌾 Trained on **real-world sensor data**
- 📈 High performance: **94% training accuracy** | **87% testing accuracy**
- 💡 **Smart decision-making** for irrigation control
- 💧 Promotes **sustainable agriculture** and **resource optimization**

---

## 🛠️ Technologies Used

- [Python](https://www.python.org/) – Programming Language  
- [TensorFlow / Keras](https://www.tensorflow.org/) – Deep Learning Framework  
- [Pandas](https://pandas.pydata.org/) – Data Manipulation  
- [NumPy](https://numpy.org/) – Numerical Computation  
- [Matplotlib](https://matplotlib.org/) – Data Visualization

---

## 🚀 Project Objective (Week-1)

> 📌 **Goal**: Develop a predictive model using ANN to classify whether irrigation is required, based on environmental data.

This sets the foundation for an intelligent, automated irrigation system that responds in real-time to weather and soil conditions.

---

## 📁 Folder Structure

```bash
Smart-Irrigation-System/
├── data/                  # Sensor dataset (CSV or preprocessed format)
├──smart_irrigation_ann         # Main code
