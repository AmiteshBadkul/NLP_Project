#!/bin/bash

# Common configuration
EPOCHS=25
BATCH_SIZE=8
LR=2e-5
WANDB_PROJECT="peft-experiments"

run_experiment() {
  local TASK=$1
  local EXTRA_ARGS=$2

  echo "Running experiments for task: $TASK"

  # No fine-tuning, BitFit, Adapter
  for STRATEGY in none bitfit adapter; do
    NAME="${TASK}_${STRATEGY}"
    python fine_tune.py \
      --task $TASK \
      --strategy $STRATEGY \
      --epochs $EPOCHS \
      --lr $LR \
      --batch_size $BATCH_SIZE \
      --save_dir $NAME \
      --wandb_project $WANDB_PROJECT \
      --wandb_run_name $NAME \
      $EXTRA_ARGS
  done

  # LoRA tuning
  for R in 4 8; do
    for ALPHA in 32; do
      for DROPOUT in 0.1; do
        NAME="${TASK}_lora_r${R}_a${ALPHA}_d${DROPOUT}"
        python fine_tune.py \
          --task $TASK \
          --strategy lora \
          --epochs $EPOCHS \
          --lr $LR \
          --batch_size $BATCH_SIZE \
          --lora_r $R \
          --lora_alpha $ALPHA \
          --lora_dropout $DROPOUT \
          --save_dir $NAME \
          --wandb_project $WANDB_PROJECT \
          --wandb_run_name $NAME \
          $EXTRA_ARGS
      done
    done
  done

  # Prefix tuning
  for PREFIX in 20; do
    NAME="${TASK}_prefix_${PREFIX}tokens"
    python fine_tune.py \
      --task $TASK \
      --strategy prefix \
      --epochs $EPOCHS \
      --lr $LR \
      --batch_size $BATCH_SIZE \
      --prefix_tokens $PREFIX \
      --save_dir $NAME \
      --wandb_project $WANDB_PROJECT \
      --wandb_run_name $NAME \
      $EXTRA_ARGS
  done
}

run_experiment "qqp" ""
run_experiment "qqp" "--small_balanced_subset"
run_experiment "mrpc" ""
