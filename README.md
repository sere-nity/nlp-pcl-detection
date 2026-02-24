# PCL Detection — SemEval-2022 Task 4

Binary classification of *Patronizing and Condescending Language* (PCL) targeting vulnerable communities, based on the Don't Patronize Me! dataset (Pérez-Almendros et al., 2020).

## Approach

Fine-tuned [`cardiffnlp/twitter-roberta-base-hate`](https://huggingface.co/cardiffnlp/twitter-roberta-base-hate) on Task 1 (binary PCL detection). The hate-speech pre-training provides a useful inductive bias since PCL shares surface features with othering language.

Training uses a downsampled 1:2 pos:neg ratio to address class imbalance, with a grid search over epochs (8/10/12), learning rate (1e-5/2e-5/3e-5), and class weight (1:1.5/2.0/3.0).

**Best config:** 10 epochs · lr=2e-5 · weight=[1.0, 3.0]

## Results (dev set, 2094 examples)

| Model | F1 | Precision | Recall |
|---|---|---|---|
| RoBERTa baseline | 0.468 | 0.344 | 0.734 |
| **BestModel** | **0.552** | **0.455** | **0.704** |

BestModel gains 8.4pp F1, primarily by improving precision — it fires less often but is more accurate when it does.

## Repo structure

```
BestModel/
  Reconstruct_and_RoBERTa_baseline_train_dev_dataset.ipynb  # training + grid search
  submission/
    dev.txt          # dev set predictions (2094 lines)
    test.txt         # test set predictions (3832 lines)
    baseline.txt     # baseline predictions for comparison

```


