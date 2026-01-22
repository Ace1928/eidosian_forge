import importlib
from functools import partial
from pathlib import Path
import torch
from llama_recipes.datasets import (
def get_preprocessed_dataset(tokenizer, dataset_config, split: str='train') -> torch.utils.data.Dataset:
    if not dataset_config.dataset in DATASET_PREPROC:
        raise NotImplementedError(f'{dataset_config.dataset} is not (yet) implemented')

    def get_split():
        return dataset_config.train_split if split == 'train' else dataset_config.test_split
    return DATASET_PREPROC[dataset_config.dataset](dataset_config, tokenizer, get_split())