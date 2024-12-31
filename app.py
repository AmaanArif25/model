import streamlit as st
import joblib
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import OneHotEncoder

# Load your trained models
rf_model = joblib.load('model/random_forest_model.sav')
lstm_model = load_model('model/lstm_model.h5')

# Create a function for k-mer prediction
def predict_next_kmer(sequence, model):
    # Generate k-mers from the input sequence (adjust for k)
    k = 4
    kmers = [sequence[i:i+k] for i in range(len(sequence) - k + 1)]
    
    # One-hot encode the k-mers
    encoder = OneHotEncoder()
    X = encoder.fit_transform(np.array(kmers).reshape(-1, 1)).toarray()
    
    # Predict the next nucleotide using the model
    next_nucleotides = model.predict(X)
    return next_nucleotides

# Streamlit app UI
st.title("Genome Scaffold Predictor")

# User input for genome sequence
user_input = st.text_area("Enter your genome sequence:", height=150)

if st.button("Predict Next K-mer"):
    if user_input:
        # Ensure valid nucleotide sequence (A, T, C, G)
        if all(base in "ATCG" for base in user_input):
            next_kmer_rf = predict_next_kmer(user_input, rf_model)
            next_kmer_lstm = predict_next_kmer(user_input, lstm_model)
            
            # Display predictions
            st.write(f"Random Forest predicted next k-mer: {next_kmer_rf}")
            st.write(f"LSTM predicted next k-mer: {next_kmer_lstm}")
        else:
            st.error("Invalid sequence. Please only enter nucleotide bases (A, T, C, G).")
    else:
        st.error("Please enter a genome sequence.")
