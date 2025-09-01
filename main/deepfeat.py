import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.svm import SVC

# Load Data
train_feat = np.load("train_feature.npz", allow_pickle=True)
valid_feat = np.load("valid_feature.npz", allow_pickle=True)
train_feat_X = train_feat['features']
train_feat_Y = train_feat['label']
valid_feat_X = valid_feat['features']

# Flatten the 13x786 matrix into vectors and standardize
scaler = StandardScaler()
train_feat_X_scaled = scaler.fit_transform(train_feat_X.reshape(len(train_feat_X), -1))
valid_feat_X_scaled = scaler.transform(valid_feat_X.reshape(len(valid_feat_X), -1))

model = SVC(kernel='rbf')

model.fit(train_feat_X_scaled, train_feat_Y)
y_pred = model.predict(valid_feat_X_scaled)
accuracy = accuracy_score(valid_feat['label'], y_pred)
print(f"SVM Model Accuracy: {accuracy:.4f}%")