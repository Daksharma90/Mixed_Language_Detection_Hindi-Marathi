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

# Correction dictionary with only lowercase keys
correction_dict = {
   # Marathi Words
    "majha": "M", "konti": "M", "boltoy": "M", "samajla": "M", "aajun": "M", 
    "shabda": "M", "aata": "M", "karaychay": "M", "zato": "M", "zate": "M", "alo": "M", "aalo": "M",
    "hotil": "M"
}

# Function to check if a word is in Devanagari script
def is_devanagari(word):
    return bool(re.search(r'[\u0900-\u097F]', word))  # Unicode range for Devanagari

# Function to classify a Romanized word using majority voting across case variations
def predict_with_variations(word, model, tokenizer):
    variations = [word.lower(), word.capitalize(), word.upper()]
    predictions = []

    for var in variations:
        inputs = tokenizer(var, return_tensors="pt", truncation=True, padding="max_length", max_length=32).to(device)
        outputs = model(**inputs)
        predicted_class = torch.argmax(outputs.logits, dim=-1).item()
        predictions.append(romanized_label_map_inv[predicted_class])

    return max(set(predictions), key=predictions.count)  # Most frequent label

# Function to classify a sentence word-wise using both models
def classify_sentence(sentence):
    words = sentence.split()
    predictions = []
    marathi_count, hindi_count = 0, 0
    
    for word in words:
        if is_devanagari(word):
            # Classify using Native Model
            inputs = native_tokenizer(word, return_tensors="pt", truncation=True, padding="max_length", max_length=32).to(device)
            outputs = native_model(**inputs)
            predicted_class = torch.argmax(outputs.logits, dim=-1).item()
            label = native_label_map_inv[predicted_class]
        else:
            # Convert to lowercase before classification
            formatted_word = word.lower()

            # Classify using Romanized Model with multiple variations
            label = predict_with_variations(formatted_word, romanized_model, romanized_tokenizer)

            # Apply correction dictionary if the word is in it
            label = correction_dict.get(formatted_word, label)

        predictions.append((word, label))

        # Count occurrences
        if label == "M":
            marathi_count += 1
        elif label == "H":
            hindi_count += 1
    
    # Total words in the sentence
    total_words = len(words)

    # Improved Sentence Classification Logic
    if total_words < 5:
        sentence_label = "M" if marathi_count > 0 else "H"
    else:
        ratio = marathi_count / total_words
        if ratio > 0.40:
            sentence_label = "M"
        elif ratio < 0.25:
            sentence_label = "H"
        else:
            sentence_label = "Mixed"

    # If sentence is mixed, calculate percentages
    hindi_percentage = round((hindi_count / total_words) * 100, 2) if sentence_label == "Mixed" else None
    marathi_percentage = round((marathi_count / total_words) * 100, 2) if sentence_label == "Mixed" else None

    return predictions, sentence_label, hindi_percentage, marathi_percentage

# Streamlit app UI
st.title("Sentence Language Classifier: Hindi or Marathi")
st.write("Enter a sentence containing Hindi, Marathi words (both Romanized and Native forms)")

# Input sentence from the user
sentence = st.text_area("Input Sentence")

# Button to classify the sentence
if st.button("Classify"):
    if sentence:
        word_predictions, sentence_language, hindi_percent, marathi_percent = classify_sentence(sentence)
        st.write(f"**Prediction:** This is a {sentence_language} sentence.")
        
        if sentence_language == "Mixed":
            st.write(f"Hindi words: {hindi_percent}%, Marathi words: {marathi_percent}%")
        
        st.write("**Word-wise predictions:**")
        for word, label in word_predictions:
            st.write(f"Word: '{word}', Predicted Label: {label}")
    else:
        st.write("Please enter a sentence to classify.")
