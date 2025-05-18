import argparse
from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM,
    Seq2SeqTrainer, Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq
)
from datasets import load_from_disk


# 🧠 Tentukan max_input_length berdasarkan model
def get_max_input_length(model_checkpoint):
    if "pegasus" in model_checkpoint:
        return 512
    elif "longformer" in model_checkpoint:
        return 4096
    else:
        return 1024


# ✨ Tokenization function with dynamic input length
def tokenize_function(example, tokenizer, max_input_length=1024):
    model_inputs = tokenizer(
        example["document"],
        max_length=max_input_length,
        padding="max_length",
        truncation=True
    )
    # 🟡 Perubahan sesuai warning Transformers v5 (tidak pakai as_target_tokenizer lagi)
    labels = tokenizer(
        text_target=example["outcome"],
        max_length=150,
        padding="max_length",
        truncation=True
    )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def main(model_checkpoint):
    print(f"🔧 Loading model & tokenizer from: {model_checkpoint}")
    model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

    print("📂 Loading preprocessed dataset...")
    dataset = load_from_disk("../data/cleaned_aclsum")

    max_input_length = get_max_input_length(model_checkpoint)
    print(f"📏 Using max_input_length: {max_input_length}")

    print("🔠 Tokenizing dataset...")
    tokenized = dataset.map(
        lambda x: tokenize_function(x, tokenizer, max_input_length=max_input_length),
        batched=True
    )

    training_args = Seq2SeqTrainingArguments(
        output_dir=f"../models/{model_checkpoint.replace('/', '_')}",
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        predict_with_generate=True,
        evaluation_strategy="epoch",
        save_total_limit=2,
        num_train_epochs=3,
        logging_dir='./logs',
        logging_steps=10,
        learning_rate=2e-5,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        tokenizer=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model)
    )

    print("🚀 Training started...")
    trainer.train()

    print(f"💾 Saving model to: ../models/{model_checkpoint.replace('/', '_')}")
    model.save_pretrained(f"../models/{model_checkpoint.replace('/', '_')}")
    tokenizer.save_pretrained(f"../models/{model_checkpoint.replace('/', '_')}")

    print("✅ Training and saving completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    args = parser.parse_args()
    main(args.model)
