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
def generate_peft_config(train_config, kwargs):
    configs = (lora_config, llama_adapter_config, prefix_config)
    peft_configs = (LoraConfig, AdaptionPromptConfig, PrefixTuningConfig)
    names = tuple((c.__name__.rstrip('_config') for c in configs))
    assert train_config.peft_method in names, f'Peft config not found: {train_config.peft_method}'
    config = configs[names.index(train_config.peft_method)]()
    update_config(config, **kwargs)
    params = asdict(config)
    peft_config = peft_configs[names.index(train_config.peft_method)](**params)
    return peft_config