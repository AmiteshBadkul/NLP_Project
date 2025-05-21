#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import time
import torch
import pandas as pd
from datasets import load_dataset, concatenate_datasets
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)
from adapters import AutoAdapterModel, AdapterConfig
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
from peft import (
    get_peft_model,
    LoraConfig,
    PrefixTuningConfig,
    TaskType
)
from transformers.trainer_utils import EvalPrediction
import wandb

def parse_args():
    parser = argparse.ArgumentParser(description="PEFT Experiment Runner")
    parser.add_argument("--task", type=str, choices=["ethos", "qqp", "sst2", "mrpc"], required=True)
    parser.add_argument("--strategy", type=str, choices=["full", "none", "bitfit", "adapter", "prefix", "lora", "dora"], required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--small_balanced_subset", action="store_true", help="Use 10k balanced QQP subset if applicable")

    # PEFT-specific args
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    parser.add_argument("--prefix_tokens", type=int, default=20)

    # WandB
    parser.add_argument("--wandb_project", type=str, default="peft-nlp")
    parser.add_argument("--wandb_run_name", type=str, default=None)

    return parser.parse_args()

def compute_metrics(p: EvalPrediction):
    preds = torch.sigmoid(torch.tensor(p.predictions)).numpy() >= 0.5
    labels = p.label_ids
    probs = torch.sigmoid(torch.tensor(p.predictions)).numpy()
    return {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall": recall_score(labels, preds, zero_division=0),
        "f1": f1_score(labels, preds, zero_division=0),
        "roc_auc": roc_auc_score(labels, probs),
        "pr_auc": average_precision_score(labels, probs)
    }

class BinaryDataCollator(DataCollatorWithPadding):
    def __call__(self, features):
        batch = super().__call__(features)
        batch["labels"] = batch["labels"].float()
        return batch

def prepare_dataset(task, args):
    if task == "ethos":
        dataset = load_dataset("ethos", "binary", trust_remote_code=True)["train"]
        dataset = dataset.train_test_split(test_size=0.3, seed=42)
        test_valid = dataset["test"].train_test_split(test_size=1/3, seed=42)
        return dataset["train"], test_valid["test"], test_valid["train"], "text"

    elif task == "qqp":
        if args.small_balanced_subset:
            full = load_dataset("glue", "qqp", split="train")
            class_0 = full.filter(lambda x: x["label"] == 0).shuffle(seed=42).select(range(5000))
            class_1 = full.filter(lambda x: x["label"] == 1).shuffle(seed=42).select(range(5000))
            balanced = concatenate_datasets([class_0, class_1]).shuffle(seed=42)
            temp = balanced.train_test_split(test_size=0.3, seed=42)
            test_valid = temp["test"].train_test_split(test_size=1/3, seed=42)
            return temp["train"], test_valid["test"], test_valid["train"], ("question1", "question2")
        else:
            glue = load_dataset("glue", "qqp")
            full = concatenate_datasets([glue["train"], glue["validation"]]).shuffle(seed=42)
            temp = full.train_test_split(test_size=0.3, seed=42)
            test_valid = temp["test"].train_test_split(test_size=1/3, seed=42)
            return temp["train"], test_valid["test"], test_valid["train"], ("question1", "question2")

    elif task == "mrpc":
        glue = load_dataset("glue", "mrpc")
        full = concatenate_datasets([glue["train"], glue["validation"]]).shuffle(seed=42)
        temp = full.train_test_split(test_size=0.3, seed=42)
        test_valid = temp["test"].train_test_split(test_size=1/3, seed=42)
        return temp["train"], test_valid["test"], test_valid["train"], ("sentence1", "sentence2")

    elif task == "sst2":
        glue = load_dataset("glue", "sst2")
        full = concatenate_datasets([glue["train"], glue["validation"]]).shuffle(seed=42)
        temp = full.train_test_split(test_size=0.3, seed=42)
        test_valid = temp["test"].train_test_split(test_size=1/3, seed=42)
        return temp["train"], test_valid["test"], test_valid["train"], "sentence"

    else:
        raise ValueError(f"Unsupported task: {task}")

def main():
    args = parse_args()
    wandb.init(project=args.wandb_project, name=args.wandb_run_name or f"{args.task}_{args.strategy}_{int(time.time())}", config=vars(args))
    task_name = args.task
    strategy = args.strategy
    output_dir = args.save_dir or f"{task_name}_{strategy}_{time.strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(output_dir, exist_ok=True)

    train_ds, test_ds, valid_ds, input_keys = prepare_dataset(task_name, args)
    model_name = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize(example):
        if isinstance(input_keys, tuple):
            return tokenizer(
                example[input_keys[0]],
                example[input_keys[1]],
                truncation=True,
                padding="max_length",
                max_length=128
            )
        return tokenizer(
            example[input_keys],
            truncation=True,
            padding="max_length",
            max_length=128
        )

    # Rename 'label' to 'labels' once, then tokenize
    if "label" in train_ds.column_names:
        train_ds = train_ds.rename_column("label", "labels")
    if "label" in valid_ds.column_names:
        valid_ds = valid_ds.rename_column("label", "labels")
    if "label" in test_ds.column_names:
        test_ds = test_ds.rename_column("label", "labels")

    train_ds = train_ds.map(tokenize)
    valid_ds = valid_ds.map(tokenize)
    test_ds = test_ds.map(tokenize)

    train_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    valid_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    test_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    base_model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1)

    peft_config = None
    if strategy == "lora":
        peft_config = LoraConfig(task_type=TaskType.SEQ_CLS, inference_mode=False, r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout)
    elif strategy == "prefix":
        peft_config = PrefixTuningConfig(task_type=TaskType.SEQ_CLS, num_virtual_tokens=args.prefix_tokens)
    elif strategy == "bitfit":
        model = base_model
        for name, param in model.named_parameters():
            param.requires_grad = False
            if ".bias" in name or "classifier" in name:
                param.requires_grad = True

    if peft_config:
        model = get_peft_model(base_model, peft_config)
    elif strategy == "full":
        model = base_model
    elif strategy == "none":
        for param in base_model.base_model.parameters():
            param.requires_grad = False
        model = base_model
    elif strategy == "dora":
        raise NotImplementedError("DoRA is not supported in Hugging Face PEFT as of now.")
    elif strategy == "adapter":
        model = AutoAdapterModel.from_pretrained(model_name)
        adapter_config = AdapterConfig.load("houlsby")
        model.add_adapter("peft_adapter", config=adapter_config)
        model.train_adapter("peft_adapter")
        model.add_classification_head("peft_adapter", num_labels=1)
        model.set_active_adapters("peft_adapter")

    training_args = TrainingArguments(
    output_dir=output_dir,
    learning_rate=args.lr,
    per_device_train_batch_size=args.batch_size,
    per_device_eval_batch_size=args.batch_size,
    num_train_epochs=args.epochs,
    eval_strategy="epoch",
    save_strategy="no",
    logging_dir=os.path.join(output_dir, "logs"),
    logging_steps=10,
    load_best_model_at_end=False,
    report_to="wandb",
    remove_unused_columns=False 
    )


    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        data_collator=BinaryDataCollator(tokenizer)
    )

    metrics_history = []
    for epoch in range(args.epochs):
        trainer.train()
        train_metrics = trainer.evaluate(train_ds)
        valid_metrics = trainer.evaluate(valid_ds)
        test_metrics = trainer.evaluate(test_ds)

        combined = {
            "epoch": epoch + 1,
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **{f"valid_{k}": v for k, v in valid_metrics.items()},
            **{f"test_{k}": v for k, v in test_metrics.items()},
        }
        wandb.log(combined)
        metrics_history.append(combined)

    df = pd.DataFrame(metrics_history)
    df.to_csv(os.path.join(output_dir, "all_epoch_metrics.csv"), index=False)
    print(df)

    model.save_pretrained(os.path.join(output_dir, "model"))
    tokenizer.save_pretrained(os.path.join(output_dir, "tokenizer"))
    wandb.finish()

if __name__ == "__main__":
    main()
