Based on the actual codebase, `run_experiments.sh`, `fine_tune.py`, plotting scripts, your environment configuration (`requirements.yml`), and the documentation in your proposal and slides, here is a complete and corrected `README.md` for your project:

---

# Minimal Effort, Maximum Effect?

## An Evaluation of Parameter-Efficient Fine-Tuning in Varying Data Regimes

**Author:** Amitesh Badkul
**Institution:** CUNY Graduate Center
**Email:** [abadkul@gradcenter.cuny.edu](mailto:abadkul@gradcenter.cuny.edu)

---

## Overview

This project evaluates the effectiveness and efficiency of various **Parameter-Efficient Fine-Tuning (PEFT)** methods when adapting BERT for downstream NLP tasks across different data availability regimes. The project analyzes the trade-off between accuracy and computational efficiency.

---

## PEFT Techniques Explored

* **LoRA (Low-Rank Adaptation)**
* **DoRA (Dynamic LoRA)** (Not Implemented Yet)
* **BitFit**
* **Prefix Tuning**
* **Adapter Tuning**
* **Full Fine-Tuning** (baseline)
* **Frozen** (no fine-tuning, control)

---

## Datasets Used

| Dataset   | Task                  | Regime                     | Samples |
| --------- | --------------------- | -------------------------- | ------- |
| **ETHOS** | Hate speech detection | Low-resource               | 1,000   |
| **MRPC**  | Paraphrase detection  | Mid-resource               | 5,000   |
| **QQP**   | Paraphrase detection  | Moderate-resource (subset) | 10,000  |

---

## Environment Setup

1. **Clone the repository:**

```bash
git clone https://github.com/AmiteshBadkul/NLP_Project.git
cd NLP_Project/environment
```

2. **Create the environment using conda:**

```bash
conda env create -f requirements.yml
conda activate adapter-peft
```

---

## Running Experiments

You can run all experiments using the bash script:

```bash
bash run_experiments.sh
```

This script executes all combinations of models and datasets using `fine_tune.py`.

### Running Individual Experiments

To run a single experiment:

```bash
python fine_tune.py --model <MODEL> --dataset <DATASET>
```

**Available model options:**

* `lora`
* `dora` (Not Implemented Yet)
* `bitfit`
* `prefix`
* `adapter`
* `full`
* `frozen`

**Available dataset options:**

* `ethos`
* `mrpc`
* `qqp`

**Example:**

```bash
python fine_tune.py --model lora --dataset ethos
```

---

## Visualization

To visualize UMAP embeddings:

```bash
python plot_umap.py --input_dir <model_output_directory>
```

To visualize attention patterns:

```bash
python plot_attention.py --input_dir <model_output_directory>
```

---


