import logging
import math
import os
from copy import deepcopy
import datasets
import evaluate
import torch
import transformers
from datasets import load_dataset
from torch.utils.data import DataLoader, IterableDataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from accelerate import Accelerator, DataLoaderConfiguration, DistributedType
from accelerate.data_loader import DataLoaderDispatcher
from accelerate.test_utils import RegressionDataset, RegressionModel, torch_device
from accelerate.utils import is_torch_xla_available, set_seed
def test_torch_metrics(accelerator: Accelerator, num_samples=82, dispatch_batches=False, split_batches=False, batch_size=16):
    _, ddp_model, dataloader = get_basic_setup(accelerator, num_samples, batch_size)
    logits, _ = generate_predictions(ddp_model, dataloader, accelerator)
    assert len(logits) == num_samples, f'Unexpected number of inputs:\n    Expected: {num_samples}\n    Actual: {len(logits)}'