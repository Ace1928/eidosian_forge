import filelock
import glob
import fnmatch
import json
import os
from collections import defaultdict
from typing import Any, Iterator, List, Optional, Tuple
from huggingface_hub import snapshot_download, HfFileSystem
import numpy as np
from safetensors.torch import load_file, save_file, safe_open
import torch
from tqdm.auto import tqdm
from vllm.config import ModelConfig
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization import (get_quantization_config,
def get_quant_config(model_config: ModelConfig) -> QuantizationConfig:
    quant_cls = get_quantization_config(model_config.quantization)
    hf_quant_config = getattr(model_config.hf_config, 'quantization_config', None)
    if hf_quant_config is not None:
        return quant_cls.from_config(hf_quant_config)
    model_name_or_path = model_config.model
    is_local = os.path.isdir(model_name_or_path)
    if not is_local:
        with get_lock(model_name_or_path, model_config.download_dir):
            hf_folder = snapshot_download(model_name_or_path, revision=model_config.revision, allow_patterns='*.json', cache_dir=model_config.download_dir, tqdm_class=Disabledtqdm)
    else:
        hf_folder = model_name_or_path
    config_files = glob.glob(os.path.join(hf_folder, '*.json'))
    quant_config_files = [f for f in config_files if any((f.endswith(x) for x in quant_cls.get_config_filenames()))]
    if len(quant_config_files) == 0:
        raise ValueError(f'Cannot find the config file for {model_config.quantization}')
    if len(quant_config_files) > 1:
        raise ValueError(f'Found multiple config files for {model_config.quantization}: {quant_config_files}')
    quant_config_file = quant_config_files[0]
    with open(quant_config_file, 'r') as f:
        config = json.load(f)
    return quant_cls.from_config(config)