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
    marathi_count, hindi_count, english_count = 0, 0, 0
    
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
        
        predictions.append((word, label))
        
        # Count occurrences
        if label == "M":
            marathi_count += 1
        elif label == "H":
            hindi_count += 1
        else:
            english_count += 1
    
    # Sentence classification based on thresholds
    total_words = len(words)
    if total_words <= 4:
        sentence_label = "M" if marathi_count >= 1 else "H"
    elif total_words <= 6:
        sentence_label = "M" if marathi_count >= 2 else "H"
    elif total_words <= 8:
        sentence_label = "M" if marathi_count >= 3 else "H"
    elif total_words <= 10:
        sentence_label = "M" if marathi_count >= 4 else "H"
    elif total_words <= 15:
        sentence_label = "M" if (marathi_count / total_words) >= 0.35 else "H"
    elif total_words <= 20:
        sentence_label = "M" if (marathi_count / total_words) >= 0.30 else "H"
    else:
        sentence_label = "M" if (marathi_count / total_words) >= 0.30 else "H"
    
    return predictions, sentence_label

# Example usage
sentence = "kya Kay"
word_predictions, sentence_language = classify_sentence(sentence)
print("Word-wise predictions:", word_predictions)
print("Predicted sentence language:", sentence_language)
