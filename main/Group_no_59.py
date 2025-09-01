import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from tensorflow.keras.layers import Input, Embedding, Flatten, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import pickle


# Utility function to save predictions to a text file
def save_predictions_to_file(predictions, filename):
    with open(filename, 'w') as f:
        for pred in predictions:
            f.write(f"{pred}\n")


# Model 1: Emoticon Model
class EmoticonModel:
    def __init__(self, vocab_size):
        """
        Initializes the EmoticonModel with an embedding layer and dense output.
        Uses an embedding layer to represent input emoticons and flattens it for classification.
        
        Parameters:
        - vocab_size: Size of the vocabulary for the embedding layer.
        """
        input_shape = (13,)  # Input sequence length
        embedding_dim = 32  # Embedding size
        
        # Define the model architecture
        self.input_layer = Input(shape=input_shape)
        self.embedding_layer = Embedding(input_dim=vocab_size, output_dim=embedding_dim)(self.input_layer)
        x = Flatten()(self.embedding_layer)
        self.output_layer = Dense(1, activation='sigmoid')(x)  # Binary classification
        self.model = Model(inputs=self.input_layer, outputs=self.output_layer)
        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    def load_weights(self, filepath):
        """
        Load the weights of the model from a specified file.
        
        Parameters:
        - filepath: Path to the weights file.
        """
        self.model.load_weights(filepath)


# Model 2: Feature-based Model using Support Vector Classifier
class FeatureModel:
    def __init__(self):
        """
        Initializes a Support Vector Classifier (SVC) with RBF kernel and a StandardScaler.
        This model is used for feature-based predictions after scaling the data.
        """
        self.model = SVC(kernel='rbf', random_state=42)  # SVC with RBF kernel
        self.scaler = StandardScaler()  # Standard Scaler for normalization

    def predict(self, X):
        """
        Predict using the trained SVC model after scaling the features.
        
        Parameters:
        - X: Feature matrix for prediction.
        """
        X_scaled = self.scaler.transform(X)  # Scale the input features
        return self.model.predict(X_scaled)

    @classmethod
    def load(cls, model_filename, scaler_filename):
        """
        Loads a pre-trained SVC model and its corresponding scaler from files.
        
        Parameters:
        - model_filename: Path to the saved SVC model.
        - scaler_filename: Path to the saved StandardScaler.
        
        Returns:
        - Instance of the FeatureModel class with the model and scaler loaded.
        """
        instance = cls()
        with open(model_filename, 'rb') as model_file:
            instance.model = pickle.load(model_file)  # Load the model
        with open(scaler_filename, 'rb') as scaler_file:
            instance.scaler = pickle.load(scaler_file)  # Load the scaler
        return instance


# Model 3: Text Sequence Model using a pre-trained GRU model
class TextSeqModel:
    def __init__(self, model_filename):
        """
        Initializes the text sequence model by loading a pre-trained GRU Based model from a .h5 file.
        
        Parameters:
        - model_filename: Path to the saved GRU Based model.
        """
        self.model = load_model(model_filename)

    def predict(self, X):
        """
        Predict using the loaded GRU model.
        
        Parameters:
        - X: Input sequence for prediction.
        
        Returns:
        - Predictions converted to binary labels.
        """
        predictions = self.model.predict(X)
        predictions = (predictions > 0.5).astype(int).flatten()  # Binary classification
        return predictions


# Model 4: Combined Model for merging predictions from different models
class CombinedModel:
    def __init__(self):
        """
        Initializes the CombinedModel by loading a pre-trained model, encoder, scaler, and PCA
        components from a saved pickle file.
        """
        with open('Combined.pkl', 'rb') as model_file:
            saved_data = pickle.load(model_file)
            self.model = saved_data['model']
            self.encoder = saved_data['encoder']
            self.scaler = saved_data['scaler']
            self.vectorizer = saved_data['vectorizer']
            self.pca = saved_data['pca']

    def predict(self, test_feat_X, test_emoticon_X, test_seq_X):
        """
        Predicts using the combined model by preprocessing and combining the features from different sources.
        
        Parameters:
        - test_feat_X: Feature data.
        - test_emoticon_X: Emoticon data.
        - test_seq_X: Sequence text data.
        
        Returns:
        - Combined model predictions.
        """
        # 1. Encode emoticons
        test_emoticons_encoded = self.encoder.transform(test_emoticon_X)

        # 2. Scale the flattened deep features
        test_features_flattened = test_feat_X.reshape(test_feat_X.shape[0], -1)
        test_features_scaled = self.scaler.transform(test_features_flattened)

        # 3. Vectorize text sequences using TF-IDF
        test_text_seq_encoded = self.vectorizer.transform(test_seq_X.flatten()).toarray()

        # Concatenate the processed features
        X_test_combined = np.concatenate([test_emoticons_encoded, test_features_scaled, test_text_seq_encoded], axis=1)

        # Apply PCA for dimensionality reduction
        X_test_combined_pca = self.pca.transform(X_test_combined)

        # Make predictions using the combined model
        predictions = self.model.predict(X_test_combined_pca)
        return predictions

    
# Main execution block
if __name__ == '__main__':
    # 1. Text Sequence Model prediction
    test_data = pd.read_csv("datasets/test/test_text_seq.csv") # Load test data
    test_seq_raw = test_data['input_str'].values  # Original text
    test_seq_X = np.array([list(map(int, s)) for s in test_data['input_str']]) # Processed text data

    text_model = TextSeqModel("text_seq.h5")  # Initialize and load model
    pred_text = text_model.predict(test_seq_X)  # Predict on text data
    save_predictions_to_file(pred_text, "pred_textseq.txt") # Save predictions to file
    print("Predictions saved to pred_textseq.txt.") # Print message

    # 2. Deep Feature Model prediction
    test_feat = np.load("datasets/test/test_feature.npz", allow_pickle=True) # Load test features
    test_feat_X = test_feat['features'] # Extract features

    feature_model = FeatureModel.load("feature_model.pkl", "feature_scaler.pkl")  # Load pre-trained model and scaler
    predictions = feature_model.predict(test_feat_X.reshape(len(test_feat_X), -1)) # Predict on test data
    save_predictions_to_file(predictions, "pred_deepfeat.txt") # Save predictions to file
    print("Predictions saved to pred_deepfeat.txt")

    # 3. Emoticon Model prediction
    with open('tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)

    test = pd.read_csv('datasets/test/test_emoticon.csv') # Load test data
    X_test = tokenizer.texts_to_sequences(test['input_emoticon']) # Tokenize emoticons
    X_test = pad_sequences(X_test, maxlen=13, padding='post') # Pad sequences

    test_emoticon_X = test['input_emoticon'].values.reshape(-1, 1) # Reshape for encoding
    vocab_size = len(tokenizer.word_index) + 1 # Vocabulary size

    emoticon_model = EmoticonModel(vocab_size) # Initialize model
    emoticon_model.load_weights('emoticon.weights.h5') # Load model weights
    predictions = emoticon_model.model.predict(X_test) # Predict on test data
    predictions = (predictions > 0.5).astype(int) # Convert to binary labels
    save_predictions_to_file(predictions.flatten(), "pred_emoticon.txt") # Save predictions to file
    print("Predictions saved to pred_emoticon.txt.")

    # 4. Combined Model prediction
    combined_model = CombinedModel() # Initialize combined model
    pred_combined = combined_model.predict(test_feat_X, test_emoticon_X, test_seq_raw) # Predict using combined model
    save_predictions_to_file(pred_combined, "pred_combined.txt") # Save predictions to file
    print("Predictions saved to pred_combined.txt.")
