import os
import torch
import argparse
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import PeftModel, PeftConfig
from datasets import load_dataset

sns.set_context("paper")
sns.set(font='serif')
sns.set_style("white", {
    "font.family": "serif",
    "font.serif": ["Times", "Palatino", "serif"]
})

plt.rcParams.update({
    'figure.dpi': 300,
})


PALETTE = {
    "Full": "#2ca02c",
    "Frozen": "#ff7f0e",
    "BitFit": "#1f77b4",
    "LoRA": "#d62728",
    "Prefix Tuning": "#9467bd"
}

SAVE_DIR = "attention_outputs"
os.makedirs(SAVE_DIR, exist_ok=True)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, required=True, choices=["qqp", "mrpc"])
    return parser.parse_args()

def get_methods(task):
    return {
        f"{task}_full": "Full",
        f"{task}_none": "Frozen",
        f"{task}_bitfit": "BitFit",
        f"{task}_lora_r4_a16_d0.1": "LoRA",
        f"{task}_prefix_10tokens": "Prefix Tuning"
    }

def load_model(folder, task):
    strategy = folder.replace(f"{task}_", "")
    model_path = os.path.join(folder, "model")
    if strategy in {"full", "none", "bitfit"}:
        model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=1, output_attentions=True)
    else:
        config = PeftConfig.from_pretrained(model_path)
        base_model = AutoModelForSequenceClassification.from_pretrained(
            config.base_model_name_or_path,
            num_labels=1,
            output_attentions=True
        )
        model = PeftModel.from_pretrained(base_model, model_path)
    tokenizer = AutoTokenizer.from_pretrained(os.path.join(folder, "tokenizer"))
    return model.eval(), tokenizer

def get_attention_scores(model, tokenizer, sentence1, sentence2=None):
    if sentence2:
        inputs = tokenizer(sentence1, sentence2, return_tensors="pt", truncation=True)
    else:
        inputs = tokenizer(sentence1, return_tensors="pt", truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    return outputs.attentions, tokens

def select_sentence(model, tokenizer, task, label=1):
    if task in ["qqp", "mrpc"]:
        dataset = load_dataset("glue", task)["train"].filter(lambda x: x["label"] == label).shuffle(seed=42).select(range(5000))
        for ex in dataset:
            s1 = ex["question1"] if task == "qqp" else ex["sentence1"]
            s2 = ex["question2"] if task == "qqp" else ex["sentence2"]
            inputs = tokenizer(s1, s2, return_tensors="pt", truncation=True)
            with torch.no_grad():
                logits = model(**inputs).logits
            pred = (logits.squeeze().sigmoid() > 0.5).long().item()
            if pred == label:
                return (s1, s2)
    else:
        raise ValueError(f"Unsupported task: {task}")
    return None

def plot_attention_arcs(attn_map, tokens, method_name, label_type):
    avg_attn = attn_map[-1][0].mean(dim=0)
    source_idx = 0

    fig, ax = plt.subplots(figsize=(len(tokens) * 0.6, 3))
    ax.set_xlim(0, len(tokens))
    ax.set_ylim(0, 1.5)
    ax.axis("off")

    for i, tok in enumerate(tokens):
        ax.text(i + 0.5, 0, tok, ha="center", va="bottom", rotation=45, fontsize=9)

    for tgt_idx in range(1, len(tokens)):
        weight = avg_attn[source_idx, tgt_idx].item()
        if weight < 0.05:
            continue
        arc = mpatches.Arc(
            ((source_idx + tgt_idx) / 2 + 0.5, 0.1),
            width=(tgt_idx - source_idx),
            height=0.5 * (tgt_idx - source_idx),
            angle=0,
            theta1=0,
            theta2=180,
            lw=2 * weight,
            alpha=0.8,
            color=PALETTE[method_name]
        )
        ax.add_patch(arc)

    filename = f"attn_arc_{method_name.replace(' ', '_')}_{label_type}.png"
    plt.title(f"{method_name} — {label_type} Sentence: [CLS] → Token Attention")
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, filename))
    plt.close()

def main():
    args = parse_args()
    task = args.task
    methods = get_methods(task)

    for folder, method_name in methods.items():
        print(f"\n== {method_name} ==")
        model, tokenizer = load_model(folder, task)

        for label_type, label_val in zip(["Positive", "Negative"], [1, 0]):
            sentence = select_sentence(model, tokenizer, task, label=label_val)
            if sentence is None:
                print(f"No correct {label_type} sentence found.")
                continue
            print(f"{label_type} sentence:\n{sentence}\n")
            attn_map, tokens = get_attention_scores(model, tokenizer, *sentence)
            plot_attention_arcs(attn_map, tokens, method_name, label_type)

if __name__ == "__main__":
    main()
