import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Embedding, Flatten, Dense, Input
from sklearn.metrics import accuracy_score

# Load the datasets
train = pd.read_csv('datasets/train/train_emoticon.csv')
valid = pd.read_csv('datasets/valid/valid_emoticon.csv')

# Tokenize the input data
tokenizer = Tokenizer(char_level=True)
tokenizer.fit_on_texts(train['input_emoticon'])

X_train = tokenizer.texts_to_sequences(train['input_emoticon'])
X_val = tokenizer.texts_to_sequences(valid['input_emoticon'])

# Padding the sequences
X_train = pad_sequences(X_train, maxlen=13, padding='post')
X_val = pad_sequences(X_val, maxlen=13, padding='post')

# Define the labels
y_train = train['label']
y_val = valid['label']

# Define the Keras model
input_shape = (13,)
embedding_dim = 32
vocab_size = len(tokenizer.word_index) + 1

# Input layer
input_layer = Input(shape=input_shape)

# Embedding layer
embedding_layer = Embedding(input_dim=vocab_size, output_dim=embedding_dim)(input_layer)

# Flatten the embeddings
x = Flatten()(embedding_layer)

# Add a Dense layer for classification
output_layer = Dense(1, activation='sigmoid')(x)

# Create the model
model = Model(inputs=input_layer, outputs=output_layer)

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model 
history = model.fit(X_train, y_train, epochs=20, validation_data=(X_val, y_val), batch_size=32)

# Evaluate the model
y_pred = model.predict(X_val)
y_pred_class = (y_pred > 0.5).astype("int32")
print('Validation Accuracy:', accuracy_score(y_val, y_pred_class))

loss, accuracy = model.evaluate(X_val, y_val)
print(f"Validation Accuracy: {accuracy * 100:.2f}%")