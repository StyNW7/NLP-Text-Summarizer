import re
import os
import string
import nltk
from datasets import load_dataset, DatasetDict
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

nltk.download("stopwords")
nltk.download("punkt")
nltk.download("wordnet")

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = word_tokenize(text)
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return " ".join(words)

def preprocess_dataset():
    raw_dataset = load_dataset("sobamchan/aclsum")
    
    def apply_cleaning(example):
        example["document"] = clean_text(example["document"])
        example["outcome"] = clean_text(example["outcome"])
        return example
    
    cleaned_dataset = raw_dataset.map(apply_cleaning)
    return cleaned_dataset

if __name__ == "__main__":
    dataset = preprocess_dataset()
    print(dataset)

    output_path = "../data/cleaned_aclsum"
    os.makedirs(output_path, exist_ok=True)
    dataset.save_to_disk(output_path)

    print(f"Dataset yang telah dibersihkan berhasil disimpan di: {output_path}")