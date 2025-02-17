import streamlit as st
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import re

# Load the models and tokenizers from Hugging Face using st.cache_resource
@st.cache_resource
def load_models():
    native_model_name = "GautamDaksh/Native_Marathi_Hindi_English_classifier"
    romanized_model_name = "GautamDaksh/Hindi-Marathi_Classifier"

    native_tokenizer = AutoTokenizer.from_pretrained(native_model_name)
    native_model = AutoModelForSequenceClassification.from_pretrained(native_model_name)

    romanized_tokenizer = AutoTokenizer.from_pretrained(romanized_model_name)
    romanized_model = AutoModelForSequenceClassification.from_pretrained(romanized_model_name)

    return native_tokenizer, native_model, romanized_tokenizer, romanized_model

# Load both models
native_tokenizer, native_model, romanized_tokenizer, romanized_model = load_models()

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
    marathi_count, hindi_count = 0, 0
    
    for word in words:
        if is_devanagari(word):
            # Classify using Native Model
            inputs = native_tokenizer(word, return_tensors="pt", truncation=True, padding="max_length", max_length=32).to(device)
            outputs = native_model(**inputs)
            predicted_class = torch.argmax(outputs.logits, dim=-1).item()
            label = native_label_map_inv[predicted_class]
        else:
            # Capitalize first letter for Romanized words
            formatted_word = word.capitalize()
            
            # Classify using Romanized Model
            inputs = romanized_tokenizer(formatted_word, return_tensors="pt", truncation=True, padding="max_length", max_length=32).to(device)
            outputs = romanized_model(**inputs)
            predicted_class = torch.argmax(outputs.logits, dim=-1).item()
            label = romanized_label_map_inv[predicted_class]
        
        # Count occurrences
        if label == "M":
            marathi_count += 1
        elif label == "H":
            hindi_count += 1
    
    # Sentence classification based on thresholds
    total_words = len(words)
    if total_words <= 4:
        sentence_label = "Marathi" if marathi_count >= 1 else "Hindi"
    elif total_words <= 6:
        sentence_label = "Marathi" if marathi_count >= 2 else "Hindi"
    elif total_words <= 8:
        sentence_label = "Marathi" if marathi_count >= 3 else "Hindi"
    elif total_words <= 10:
        sentence_label = "Marathi" if marathi_count >= 4 else "Hindi"
    elif total_words <= 15:
        sentence_label = "Marathi" if (marathi_count / total_words) >= 0.35 else "Hindi"
    elif total_words <= 20:
        sentence_label = "Marathi" if (marathi_count / total_words) >= 0.30 else "Hindi"
    else:
        sentence_label = "Marathi" if (marathi_count / total_words) >= 0.30 else "Hindi"
    
    return sentence_label

# Streamlit app UI
st.title("Sentence Language Classifier: Hindi or Marathi")
st.write("Enter a sentence containing Hindi, Marathi words (both Romanized and Native forms)")

# Input sentence from the user
sentence = st.text_area("Input Sentence")

# Button to classify the sentence
if st.button("Classify"):
    if sentence:
        sentence_language = classify_sentence(sentence)
        st.write(f"**Prediction:** This is a {sentence_language} sentence.")
    else:
        st.write("Please enter a sentence to classify.")
