import argparse
from transformers import (AutoTokenizer, AutoModelForSeq2SeqLM,
                          Seq2SeqTrainer, Seq2SeqTrainingArguments,
                          DataCollatorForSeq2Seq)
from datasets import load_dataset, load_from_disk

def tokenize_function(example, tokenizer):
    model_inputs = tokenizer(example["document"], max_length=1024, padding="max_length", truncation=True)
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(example["outcome"], max_length=150, padding="max_length", truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def main(model_checkpoint):

    model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

    # Original Dataset
    # dataset = load_dataset("sobamchan/aclsum")

    # Preprocess Dataset
    dataset = load_from_disk("../data/cleaned_aclsum")
    
    tokenized = dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True)
    
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

    trainer.train()
    model.save_pretrained(f"../models/{model_checkpoint.replace('/', '_')}")
    tokenizer.save_pretrained(f"../models/{model_checkpoint.replace('/', '_')}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    args = parser.parse_args()
    main(args.model)