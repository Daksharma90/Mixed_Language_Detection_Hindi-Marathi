import streamlit as st
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
from torch.nn.functional import softmax
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

# Function to predict language for a single word
def predict_language(word, is_native):
    tokenizer = native_tokenizer if is_native else romanized_tokenizer
    model = native_model if is_native else romanized_model

    inputs = tokenizer(word, return_tensors="pt", truncation=True, padding="max_length", max_length=32).to(device)
    outputs = model(**inputs)
    logits = outputs.logits
    probabilities = softmax(logits, dim=1)
    predicted_class = torch.argmax(probabilities, dim=1).item()

    label_map = native_label_map_inv if is_native else romanized_label_map_inv
    return label_map[predicted_class]

# Function to classify a sentence word-wise using both models
def classify_sentence(sentence):
    words = sentence.split()
    predictions = []

    for word in words:
        is_native = is_devanagari(word)  # Check if the word is native or romanized
        word = word.capitalize() if not is_native else word  # Capitalize first letter if romanized
        label = predict_language(word, is_native)
        predictions.append((word, label))

    return predictions

# Calculate percentage of Hindi and Marathi words
def calculate_percentages(labeled_words):
    total_words = len(labeled_words)
    hindi_count = sum(1 for _, label in labeled_words if label == 'H')
    marathi_count = total_words - hindi_count
    hindi_percentage = (hindi_count / total_words) * 100 if total_words > 0 else 0
    marathi_percentage = (marathi_count / total_words) * 100 if total_words > 0 else 0
    return hindi_percentage, marathi_percentage

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

        st.subheader("Language Percentages:")
        hindi_percentage, marathi_percentage = calculate_percentages(result)
        st.write(f"**Hindi:** {hindi_percentage:.2f}%")
        st.write(f"**Marathi:** {marathi_percentage:.2f}%")
    else:
        st.write("Please enter a sentence to classify.")
