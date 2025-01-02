# 

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

## Quick start

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

# run lm eval with puntma-axiom
export CUDA_VISIBLE_DEVICES=0
lm_eval --model vllm \
    --model_args ${model_args} \
    --tasks ${benchmark_and_optional_task} \
    --trust_remote_code \
    --batch_size auto:4 \
    --device cuda \
    --output_path ${model_task_output_path} \
    --log_samples

```