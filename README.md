# Putnam-AXIOM Preview Evaluation Guide

## Install
```bash
conda create -n putnam_axiom python=3.11
conda activate putnam_axiom

cd ~/putnam-axiom-preview-lm-evaluation-harness

# Instead of the usual pip install lm_eval[vllm] lm-harness recommends, do the command bellow:
pip install -e ".[vllm]"
pip install antlr4-python3-runtime==4.11
# to check installs worked do (versions and paths should appear)
pip list | grep lm_eval
pip list | grep vllm
pip list | grep antlr4
```

## Quick start with CLI/BASH

Evaluate Gemma-2-2b on the original 53 Putnam questions:
```bash
# activate py env
conda activate putnam_axiom

# model & model args
export model_name_or_path="google/gemma-2-2b"
export model_args="pretrained=${model_name_or_path},dtype=auto"

# select putnam-axiom benchmark
export benchmark_and_optional_task='putnam_axiom_53'

# saving outputs
export model_task_output_path='$HOME/data/runs_outs'
mkdir -p $model_task_output_path

# choose gpu
export CUDA_VISIBLE_DEVICES=3

# run lm eval with puntma-axiom
lm_eval --model vllm \
    --model_args ${model_args} \
    --tasks ${benchmark_and_optional_task} \
    --trust_remote_code \
    --batch_size auto:4 \
    --device cuda \
    --output_path ${model_task_output_path} \
    --log_samples

```
Sample output:
```bash
|     Tasks     |Version|Filter|n-shot|  Metric   |   |Value |   |Stderr|
|---------------|------:|------|-----:|-----------|---|-----:|---|-----:|
|putnam_axiom_53|      1|none  |     4|exact_match|↑  |0.0189|±  |0.0189|
```

Evaluate Gemma-2-2b on the 265 (5*53) Variations Putnam problems:
```bash
# activate py env
conda activate putnam_axiom

# model & model args
export model_name_or_path="google/gemma-2-2b"
export model_args="pretrained=${model_name_or_path},dtype=auto"

# select putnam-axiom benchmark
export benchmark_and_optional_task='putnam_axiom_variations'

# saving outputs
export model_task_output_path='$HOME/data/runs_outs'
mkdir -p $model_task_output_path

# choose gpu
export CUDA_VISIBLE_DEVICES=3

# run lm eval with puntma-axiom
lm_eval --model vllm \
    --model_args ${model_args} \
    --tasks ${benchmark_and_optional_task} \
    --trust_remote_code \
    --batch_size auto:4 \
    --device cuda \
    --output_path ${model_task_output_path} \
    --log_samples

```
Sample output:
```bash
|         Tasks         |Version|Filter|n-shot|  Metric   |   |Value|   |Stderr|
|-----------------------|------:|------|-----:|-----------|---|----:|---|-----:|
|putnam_axiom_variations|      1|none  |     4|exact_match|↑  |    0|±  |     0|
```

note: use `putnam_axiom_original` for `benchmark_and_optional_task` to use the the original `256` problems used for the original evaluation. 

## Quick start with Pythono
Evluate Gemma-2-2b on the original 53 Putnam questions:
```bash
# set python env
conda activate putnam_axiom

# get desired model to evaluate
export model_name_or_path='google/gemma-2-2b'

# putnam examiom benchmark
export task="putnam_axiom_53"

# wandb run "off"
export mode='dryrun'

# run python eval code
python ~/putnam-axiom-preview-lm-evaluation-harness/lm_eval_py_run.py --task ${task} --model_name_or_path ${model_name_or_path} --mode ${mode}
```
Sample output:
```bash
wandb: Run summary:
wandb:       eval_bench/putnam_axiom_53/accuracy 0.01887
wandb: eval_bench/putnam_axiom_53/checkpoint_idx google/gemma-2-2b
```

Evluate Gemma-2-2b on the 265 (53 * 5) Variations Putnam questions:
```bash
# set python env
conda activate putnam_axiom

# get desired model to evaluate
export model_name_or_path='google/gemma-2-2b'

# putnam examiom benchmark
export task="putnam_axiom_53"

# wandb run "off"
export mode='dryrun'

# run python eval code
python ~/brando9/putnam-axiom-preview-lm-evaluation-harness/lm_eval_py_run.py --task ${task} --model_name_or_path ${model_name_or_path} --mode ${mode}
```

Sample output:
```bash

```

Note: change `MyLM`'s `generate_until` to call OpenAI's api (or desired closed model) to evaluate a closed model and set key. 
