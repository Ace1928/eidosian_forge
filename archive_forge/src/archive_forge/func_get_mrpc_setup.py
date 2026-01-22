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
def get_mrpc_setup(dispatch_batches, split_batches):
    dataloader_config = DataLoaderConfiguration(dispatch_batches=dispatch_batches, split_batches=split_batches)
    accelerator = Accelerator(dataloader_config=dataloader_config)
    dataloader = get_dataloader(accelerator, not dispatch_batches)
    model = AutoModelForSequenceClassification.from_pretrained('hf-internal-testing/mrpc-bert-base-cased', return_dict=True)
    ddp_model, ddp_dataloader = accelerator.prepare(model, dataloader)
    return ({'ddp': [ddp_model, ddp_dataloader, torch_device], 'no': [model, dataloader, accelerator.device]}, accelerator)