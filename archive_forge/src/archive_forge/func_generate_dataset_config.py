import inspect
from dataclasses import asdict
import torch.distributed as dist
from torch.utils.data import DistributedSampler
from peft import (
from transformers import default_data_collator
from transformers.data import DataCollatorForSeq2Seq
from llama_recipes.configs import datasets, lora_config, llama_adapter_config, prefix_config, train_config
from llama_recipes.data.sampler import LengthBasedBatchSampler, DistributedLengthBasedBatchSampler
from llama_recipes.utils.dataset_utils import DATASET_PREPROC
def generate_dataset_config(train_config, kwargs):
    names = tuple(DATASET_PREPROC.keys())
    assert train_config.dataset in names, f'Unknown dataset: {train_config.dataset}'
    dataset_config = {k: v for k, v in inspect.getmembers(datasets)}[train_config.dataset]()
    update_config(dataset_config, **kwargs)
    return dataset_config