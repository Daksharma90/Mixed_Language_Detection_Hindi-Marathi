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

# Correction dictionary with only lowercase keys
correction_dict = {
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
    marathi_count, hindi_count, english_count = 0, 0, 0

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

        # Count occurrences
        if label == "M":
            marathi_count += 1
        elif label == "H":
            hindi_count += 1
        else:
            english_count += 1

    # Total words in the sentence
    total_words = len(words)

    # Sentence Classification Logic
    if total_words < 5:
        sentence_label = "Marathi" if marathi_count > 0 else "Hindi"
    else:
        ratio = marathi_count / total_words
        if ratio > 0.40:
            sentence_label = "Marathi"
        elif ratio < 0.25:
            sentence_label = "Hindi"
        else:
            sentence_label = "Mixed"

    # If sentence is mixed, calculate percentages
    hindi_percentage = round((hindi_count / total_words) * 100, 2) if sentence_label == "Mixed" else None
    marathi_percentage = round((marathi_count / total_words) * 100, 2) if sentence_label == "Mixed" else None

    return sentence_label, hindi_percentage, marathi_percentage

# Streamlit App Setup
st.title("Sentence Language Classifier")
st.write("Enter a sentence in Romanized or native Hindi/Marathi/English")

# Text input for the user
sentence_input = st.text_input("Enter sentence:")

if sentence_input:
    # Classify the sentence
    sentence_language, hindi_percent, marathi_percent = classify_sentence(sentence_input)

    # Display sentence language prediction
    if sentence_language == "Mixed":
        st.write(f"This is a Mixed Sentence.")
        st.write(f"Hindi words: {hindi_percent}%, Marathi words: {marathi_percent}%")
    else:
        st.write(f"This is a {sentence_language} Sentence.")
