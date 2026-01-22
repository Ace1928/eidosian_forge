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
def prepare_hf_model_weights(model_name_or_path: str, cache_dir: Optional[str]=None, load_format: str='auto', fall_back_to_pt: bool=True, revision: Optional[str]=None) -> Tuple[str, List[str], bool]:
    is_local = os.path.isdir(model_name_or_path)
    use_safetensors = False
    if load_format == 'auto':
        allow_patterns = ['*.safetensors', '*.bin']
    elif load_format == 'safetensors':
        use_safetensors = True
        allow_patterns = ['*.safetensors']
    elif load_format == 'pt':
        allow_patterns = ['*.pt']
    elif load_format == 'npcache':
        allow_patterns = ['*.bin']
    else:
        raise ValueError(f'Unknown load_format: {load_format}')
    if fall_back_to_pt:
        allow_patterns += ['*.pt']
    if not is_local:
        fs = HfFileSystem()
        file_list = fs.ls(model_name_or_path, detail=False, revision=revision)
        for pattern in allow_patterns:
            matching = fnmatch.filter(file_list, pattern)
            if len(matching) > 0:
                allow_patterns = [pattern]
                break
        logger.info(f'Using model weights format {allow_patterns}')
        with get_lock(model_name_or_path, cache_dir):
            hf_folder = snapshot_download(model_name_or_path, allow_patterns=allow_patterns, cache_dir=cache_dir, tqdm_class=Disabledtqdm, revision=revision)
    else:
        hf_folder = model_name_or_path
    hf_weights_files: List[str] = []
    for pattern in allow_patterns:
        hf_weights_files += glob.glob(os.path.join(hf_folder, pattern))
        if len(hf_weights_files) > 0:
            if pattern == '*.safetensors':
                use_safetensors = True
            break
    if not use_safetensors:
        blacklist = ['training_args.bin', 'optimizer.bin', 'optimizer.pt', 'scheduler.pt', 'scaler.pt']
        hf_weights_files = [f for f in hf_weights_files if not any((f.endswith(x) for x in blacklist))]
    if len(hf_weights_files) == 0:
        raise RuntimeError(f'Cannot find any model weights with `{model_name_or_path}`')
    return (hf_folder, hf_weights_files, use_safetensors)