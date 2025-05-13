# Putnam AXIOM ZIP-FIT Benchmark

This benchmark contains the Putnam AXIOM dataset split according to the ZIP-FIT methodology, as described in the paper [ZIP-FIT: Embedding-Free Data Selection via Compression-Based Alignment](https://arxiv.org/abs/2410.18194).

## Dataset

The dataset is hosted at [zipfit/Putnam-AXIOM-for-zip-fit-splits](https://huggingface.co/datasets/zipfit/Putnam-AXIOM-for-zip-fit-splits) and contains Putnam math competition problems with the following splits:

- **Train**: 150 examples
- **Validation**: 150 examples
- **Test**: 222 examples

These splits are derived from the original 522 Putnam problems found in the main Putnam-AXIOM repository.

## Tasks

There are three individual tasks in this benchmark:

- `putnam_axiom_zipfit_train`: Evaluates models on the training split
- `putnam_axiom_zipfit_val`: Evaluates models on the validation split
- `putnam_axiom_zipfit_test`: Evaluates models on the test split

Additionally, there is a group task that combines all three:

- `putnam_axiom_zipfit_all`: Runs evaluation on all splits and reports aggregate results

## Usage

To run evaluation on a specific split:

```bash
lm_eval --model MODEL_NAME \
    --model_args MODEL_ARGS \
    --tasks putnam_axiom_zipfit_test \
    --device cuda \
    --output_path results.json
```

To evaluate on all splits:

```bash
lm_eval --model MODEL_NAME \
    --model_args MODEL_ARGS \
    --tasks putnam_axiom_zipfit_all \
    --device cuda \
    --output_path results.json
```

## Citation

```bibtex
@article{putnam_axiom2025,
  title={Putnam-AXIOM: A Functional and Static Benchmark for Measuring Higher Level Mathematical Reasoning},
  author={Aryan Gulati and Brando Miranda and Eric Chen and Emily Xia and Kai Fronsdal and Bruno de Moraes Dumont and Sanmi Koyejo},
  journal={39th International Conference on Machine Learning (ICML 2025)},
  year={2025},
  note={Preprint available at: https://openreview.net/pdf?id=YXnwlZe0yf, ICML paper: https://openreview.net/forum?id=kqj2Cn3Sxr}
}

@article{miranda2024zipfit,
  title={ZIP-FIT: Embedding-Free Data Selection via Compression-Based Alignment},
  year = {2024},
  journal = {arXiv preprint arXiv:2410.18194},
}
``` 