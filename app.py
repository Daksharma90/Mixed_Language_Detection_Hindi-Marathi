import streamlit as st
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import re

# Load the models and tokenizers from Hugging Face
native_model_name = "GautamDaksh/Native_Marathi_Hindi_English_classifier"
romanized_model_name = "GautamDaksh/Hindi-Marathi_Classifier"

native_tokenizer = AutoTokenizer.from_pretrained(native_model_name)
native_model = AutoModelForSequenceClassification.from_pretrained(native_model_name)

romanized_tokenizer = AutoTokenizer.from_pretrained(romanized_model_name)
romanized_model = AutoModelForSequenceClassification.from_pretrained(romanized_model_name)

# Move models to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
native_model.to(device)
romanized_model.to(device)

# Reverse label mapping
native_label_map_inv = {0: "E", 1: "H", 2: "M"}  # English, Hindi, Marathi for native script
romanized_label_map_inv = {0: "H", 1: "M"}  # Hindi, Marathi for Romanized script

# Function to check if a word is in Devanagari script
def is_devanagari(word):
    return bool(re.search(r'[\u0900-\u097F]', word))  # Unicode range for Devanagari

# Function to classify a sentence word-wise using both models
def classify_sentence(sentence):
    words = sentence.split()
    predictions = []

    for word in words:
        if is_devanagari(word):
            # Classify using Native Model
            inputs = native_tokenizer(word, return_tensors="pt", truncation=True, padding="max_length", max_length=32).to(device)
            outputs = native_model(**inputs)
            predicted_class = torch.argmax(outputs.logits, dim=-1).item()
            predictions.append((word, native_label_map_inv[predicted_class]))
        else:
            # Capitalize first letter for Romanized words
            formatted_word = word.capitalize()
            
            # Classify using Romanized Model
            inputs = romanized_tokenizer(formatted_word, return_tensors="pt", truncation=True, padding="max_length", max_length=32).to(device)
            outputs = romanized_model(**inputs)
            predicted_class = torch.argmax(outputs.logits, dim=-1).item()
            predictions.append((word, romanized_label_map_inv[predicted_class]))

    return predictions

# Streamlit app UI
st.title("Word-Wise Classification of Hindi, Marathi, and English Words")
st.write("Enter a sentence containing Hindi, Marathi, and English words (both Romanized and Native forms)")

# Input sentence from the user
sentence = st.text_area("Input Sentence", "Aaj office nahi jau शकत कारण मी busy आहे")

# Button to classify the sentence
if st.button("Classify"):
    if sentence:
        result = classify_sentence(sentence)
        st.write("**Classification Results**:")
        for word, label in result:
            st.write(f"Word: {word}, Label: {label}")
    else:
        st.write("Please enter a sentence to classify.")
