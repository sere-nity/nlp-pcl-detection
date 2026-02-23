# -*- coding: utf-8 -*-
"""
best_model.py
NLP Coursework 2026 - PCL Detection
Fine-tuned model for SemEval 2022 Task 4 (Subtask 1): Binary PCL Classification
"""

# ============================================================
# 1. IMPORTS AND SETUP
# ============================================================
import os
import logging
import random
import torch
import pandas as pd
from urllib import request
from collections import Counter
from ast import literal_eval
from simpletransformers.classification import ClassificationModel, ClassificationArgs

# Prepare logger
logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

# Check GPU
cuda_available = torch.cuda.is_available()
print(f"CUDA available: {cuda_available}")
if cuda_available:
    print(f"GPU: {torch.cuda.get_device_name(0)}")
else:
    print("WARNING: No GPU found, training will be slow.")

# ============================================================
# 2. HELPER FUNCTIONS
# ============================================================
def labels2file(p, outf_path):
    """Save predictions to output file, one prediction per line."""
    with open(outf_path, 'w') as outf:
        for pi in p:
            outf.write(','.join([str(k) for k in pi]) + '\n')

# ============================================================
# 3. FETCH DONT PATRONIZE ME DATA MODULE
# ============================================================
module_url = "https://raw.githubusercontent.com/Perez-AlmendrosC/dontpatronizeme/master/semeval-2022/dont_patronize_me.py"
module_name = module_url.split('/')[-1]

if not os.path.exists(module_name):
    print(f"Fetching {module_url}")
    with request.urlopen(module_url) as f, open(module_name, 'w') as outf:
        outf.write(f.read().decode('utf-8'))
else:
    print(f"{module_name} already exists, skipping download.")

from dont_patronize_me import DontPatronizeMe

# ============================================================
# 4. LOAD DATA
# ============================================================
dpm = DontPatronizeMe('.', '.')
dpm.load_task1()
dpm.load_task2(return_one_hot=True)

# Load paragraph IDs for train and dev splits
trids = pd.read_csv('train_semeval_parids-labels.csv')
teids = pd.read_csv('dev_semeval_parids-labels.csv')

trids.par_id = trids.par_id.astype(str)
teids.par_id = teids.par_id.astype(str)

data = dpm.train_task1_df

# ============================================================
# 5. REBUILD TRAINING SET (Task 1)
# ============================================================
print("Building training set...")
rows = []
for idx in range(len(trids)):
    parid = trids.par_id[idx]
    keyword = data.loc[data.par_id == parid].keyword.values[0]
    text = data.loc[data.par_id == parid].text.values[0]
    label = data.loc[data.par_id == parid].label.values[0]
    rows.append({
        'par_id': parid,
        'community': keyword,
        'text': text,
        'label': label
    })

trdf1 = pd.DataFrame(rows)
print(f"Training set size: {len(trdf1)} | Label distribution:\n{trdf1.label.value_counts()}")

# ============================================================
# 6. REBUILD DEV SET (Task 1)
# ============================================================
print("Building dev set...")
rows = []
for idx in range(len(teids)):
    parid = teids.par_id[idx]
    keyword = data.loc[data.par_id == parid].keyword.values[0]
    text = data.loc[data.par_id == parid].text.values[0]
    label = data.loc[data.par_id == parid].label.values[0]
    rows.append({
        'par_id': parid,
        'community': keyword,
        'text': text,
        'label': label
    })

tedf1 = pd.DataFrame(rows)
print(f"Dev set size: {len(tedf1)} | Label distribution:\n{tedf1.label.value_counts()}")

# ============================================================
# 7. DOWNSAMPLE NEGATIVE INSTANCES FOR TRAINING
# ============================================================
pcldf = trdf1[trdf1.label == 1]
npos = len(pcldf)
training_set1 = pd.concat([pcldf, trdf1[trdf1.label == 0][:npos * 2]])
training_set1 = training_set1.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle
print(f"Downsampled training set: {len(training_set1)} | PCL: {npos} | No-PCL: {npos*2}")

# ============================================================
# 8. YOUR BEST MODEL GOES BELOW THIS LINE
# ============================================================