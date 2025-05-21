import os
import argparse
import torch
import umap
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel, PeftConfig
from tqdm import tqdm

sns.set_context("paper")
sns.set(font='serif')
sns.set_style("white", {
    "font.family": "serif",
    "font.serif": ["Times", "Palatino", "serif"]
})

plt.rcParams.update({'figure.dpi': 300})

PALETTE = {
    "Full": "#2ca02c",
    "Frozen": "#ff7f0e",
    "BitFit": "#1f77b4",
    "LoRA": "#d62728",
    "Prefix Tuning": "#9467bd"
}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, required=True, choices=["ethos", "qqp", "mrpc"])
    return parser.parse_args()

def get_methods(task):
    return {
        f"{task}_full": "Full",
        f"{task}_none": "Frozen",
        f"{task}_bitfit": "BitFit",
        f"{task}_lora_r4_a16_d0.1": "LoRA",
        f"{task}_prefix_10tokens": "Prefix Tuning"
    }

def load_subset(task, tokenizer, n=100):
    if task == "ethos":
        ds = load_dataset("ethos", "binary", trust_remote_code=True)["train"]
        ds = ds.shuffle(seed=42).select(range(n))
        def tokenize(example):
            return tokenizer(example["text"], truncation=True, padding="max_length", max_length=128)
    elif task in ["qqp", "mrpc"]:
        glue = load_dataset("glue", task)["train"]
        ds = glue.filter(lambda x: x["label"] in [0, 1]).shuffle(seed=42).select(range(n))
        key1 = "question1" if task == "qqp" else "sentence1"
        key2 = "question2" if task == "qqp" else "sentence2"
        def tokenize(example):
            return tokenizer(example[key1], example[key2], truncation=True, padding="max_length", max_length=128)
    else:
        raise ValueError(f"Unsupported task: {task}")

    ds = ds.map(tokenize)
    ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    return ds

def load_model(folder, task):
    strategy = folder.replace(f"{task}_", "")
    model_path = os.path.join(folder, "model")

    if strategy in {"full", "none", "bitfit"}:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_path, num_labels=1, output_hidden_states=True)
    else:
        config = PeftConfig.from_pretrained(model_path)
        base_model = AutoModelForSequenceClassification.from_pretrained(
            config.base_model_name_or_path, num_labels=1, output_hidden_states=True)
        model = PeftModel.from_pretrained(base_model, model_path)

    tokenizer = AutoTokenizer.from_pretrained(os.path.join(folder, "tokenizer"))
    return model.eval(), tokenizer

@torch.no_grad()
def extract_cls_embeddings(model, dataloader, device):
    model.to(device)
    cls_embeddings, labels = [], []

    for batch in tqdm(dataloader, desc="Extracting embeddings"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = outputs.hidden_states[-1]
        cls = last_hidden[:, 0, :]
        cls_embeddings.append(cls.cpu())
        labels.extend(batch["label"].tolist())

    return torch.cat(cls_embeddings).numpy(), labels

def main():
    import torch.utils.data as data
    args = parse_args()
    task = args.task
    methods = get_methods(task)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    all_points = []
    for folder, label in methods.items():
        model, tokenizer = load_model(folder, task)
        dataset = load_subset(task, tokenizer, n=100)
        dataloader = data.DataLoader(dataset, batch_size=8)
        cls_vecs, labels = extract_cls_embeddings(model, dataloader, device)

        df = pd.DataFrame(cls_vecs)
        df["label"] = labels
        df["method"] = label
        all_points.append(df)

    df_all = pd.concat(all_points, ignore_index=True)

    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="cosine", random_state=42)
    embedding = reducer.fit_transform(df_all.drop(columns=["label", "method"]))
    df_all["UMAP1"] = embedding[:, 0]
    df_all["UMAP2"] = embedding[:, 1]

    plt.figure(figsize=(10, 7))
    sns.scatterplot(
        data=df_all,
        x="UMAP1", y="UMAP2",
        hue="method",
        style="label",
        palette=PALETTE,
        s=70,
        alpha=0.9
    )
    plt.title(f"UMAP of CLS Embeddings — {task.upper()}")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.)
    plt.tight_layout()
    plt.savefig(f"umap_{task}.png")

if __name__ == "__main__":
    main()
