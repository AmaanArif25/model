import streamlit as st
import joblib
from tensorflow.keras.models import load_model
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import pandas as pd

# Load models
rf_model = joblib.load('random_forest_model.sav')  # Load Random Forest model
lstm_model = load_model('lstm_model.h5')  # Load LSTM model

# Load k-mer encoder
kmer_mapping_df = pd.read_csv("kmer_mapping.csv")
encoder = OneHotEncoder()
encoder.fit(kmer_mapping_df["k-mer"].values.reshape(-1, 1))

def generate_kmers(sequence, k=4):
    """
    Generate non-overlapping k-mers from a genome sequence.
    """
    return [sequence[i:i + k] for i in range(0, len(sequence), k) if len(sequence[i:i + k]) == k]

def predict_next_nucleotide_rf(kmer):
    """
    Predict the next nucleotide using Random Forest model.
    """
    kmer_encoded = encoder.transform([[kmer]]).toarray()
    prediction = rf_model.predict(kmer_encoded)
    nucleotide_map = kmer_mapping_df["Next Nucleotide"].astype('category').cat.categories
    return nucleotide_map[prediction[0]]

def predict_next_nucleotide_lstm(kmer):
    """
    Predict the next nucleotide using LSTM model.
    """
    kmer_encoded = encoder.transform([[kmer]]).toarray()
    kmer_encoded = np.expand_dims(kmer_encoded, axis=1)  # Reshape for LSTM input
    prediction = lstm_model.predict(kmer_encoded)
    nucleotide_map = kmer_mapping_df["Next Nucleotide"].astype('category').cat.categories
    return nucleotide_map[np.argmax(prediction)]

# Streamlit App
st.title("Advanced Genome Scaffolding Tool")

# Input Section
st.header("Input Genome Sequence")
sequence_input = st.text_area("Enter your genome sequence (A, T, C, G):", "")
kmer_size = st.number_input("K-mer size:", min_value=2, max_value=10, value=4)

# Validate Input
if st.button("Validate Sequence"):
    if set(sequence_input.upper()).issubset({'A', 'T', 'C', 'G'}):
        st.success("Valid nucleotide sequence!")
    else:
        st.error("Invalid sequence. Please enter only A, T, C, G.")

# Generate K-mers
if sequence_input and st.button("Generate K-mers"):
    kmers = generate_kmers(sequence_input.upper(), k=kmer_size)
    st.write("Generated K-mers:", kmers)

# Model Predictions
st.header("Predict Next Nucleotide")
selected_kmer = st.selectbox("Select a K-mer for prediction:", kmers if sequence_input else [])
if selected_kmer and st.button("Predict Next Nucleotide"):
    rf_prediction = predict_next_nucleotide_rf(selected_kmer)
    lstm_prediction = predict_next_nucleotide_lstm(selected_kmer)
    st.write(f"Random Forest Prediction: {rf_prediction}")
    st.write(f"LSTM Prediction: {lstm_prediction}")
