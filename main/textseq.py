import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Embedding, Dropout
from tensorflow.keras.optimizers import Adam

# Load the training data
data = pd.read_csv("datasets/train/train_text_seq.csv")
print(data.shape)

# Preprocess the training data
X = np.array([list(map(int, s)) for s in data['input_str']])  # Shape: (num_samples, 50)
y = data['label'].astype(int).values  # Convert labels to integers

# GRU-based model for text sequence classification
model = Sequential()
model.add(Embedding(input_dim=10, output_dim=6, input_length=50))  # Embedding output_dim to 6
model.add(GRU(16, return_sequences=True))  # GRU layer with 16 units
model.add(Dropout(0.4))  # Dropout layer to prevent overfitting (adjusted dropout rate)
model.add(GRU(32))  # Another GRU layer with 32 units
model.add(Dense(1, activation='sigmoid'))  # Binary classification

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.005), loss='binary_crossentropy', metrics=['accuracy'])

# Train the model on the entire training dataset
history = model.fit(X, y, epochs=30, batch_size=16)

# Get total number of parameters
total_params = model.count_params()
print(f"Total number of parameters: {total_params}")
model.summary()

# Load the external validation dataset
valid_data = pd.read_csv("datasets/valid/valid_text_seq.csv")
print(valid_data.shape)

# Preprocess the validation data
X_test = np.array([list(map(int, s)) for s in valid_data['input_str']])  # Shape: (num_samples, 50)
y_test = valid_data['label'].astype(int).values  # Convert labels to integers

# Evaluate the model on the external validation dataset
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Validation Dataset Accuracy: {test_acc*100:.2f}%")
