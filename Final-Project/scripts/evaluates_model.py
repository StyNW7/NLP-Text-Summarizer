import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from evaluate import load as load_metric

def get_max_input_length(model_name):
    if "pegasus" in model_name:
        return 512
    elif "longformer" in model_name:
        return 4096
    else:
        return 1024

def generate_summaries(model_name, dataset_split="test"):
    model_dir = f"../models/{model_name.replace('/', '_')}"
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir).to("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    dataset = load_from_disk("../data/cleaned_aclsum")[dataset_split].select(range(10))

    predictions = []
    references = []
    max_len = get_max_input_length(model_name)

    print(f"⏳ Generating summaries for {model_name}...")
    for entry in dataset:
        if entry["document"].strip() == "" or entry["outcome"].strip() == "":
            continue
        inputs = tokenizer(entry["document"], return_tensors="pt", truncation=True, max_length=max_len).to(model.device)
        summary_ids = model.generate(inputs.input_ids, max_length=150, num_beams=4)
        pred = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        predictions.append(pred)
        references.append(entry["outcome"])

    return predictions, references

def compute_metrics(predictions, references):
    rouge = load_metric("rouge")
    result = rouge.compute(predictions=predictions, references=references, use_stemmer=True)
    return {
        "ROUGE-1": result["rouge1"],
        "ROUGE-2": result["rouge2"],
        "ROUGE-L": result["rougeL"]
    }

def main():
    models = [
        "t5-small",
        "facebook/bart-base",
        "google/pegasus-xsum",
        "allenai/led-base-16384"
    ]
    rows = []
    for m in models:
        preds, refs = generate_summaries(m)
        metrics = compute_metrics(preds, refs)
        metrics["Model"] = m
        rows.append(metrics)

    df = pd.DataFrame(rows)
    df = df[["Model", "ROUGE-1", "ROUGE-2", "ROUGE-L"]]

    # Save
    os.makedirs("../results", exist_ok=True)
    df.to_csv("../results/model_comparison.csv", index=False)

    # Visualize
    print("\n📊 Results of Model Comparison:")
    print(df)

    df.set_index("Model").plot(kind="bar", figsize=(10, 6))
    plt.title("Model Summarization Comparison (ROUGE Scores)")
    plt.ylabel("Score")
    plt.ylim(0.0, 1.0)
    plt.tight_layout()
    plt.savefig("../results/model_comparison.png")
    plt.close()
    print("✅ Graph save at: results/model_comparison.png")

if __name__ == "__main__":
    main()
