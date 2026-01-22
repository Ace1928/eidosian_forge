import json
import os
from typing import List, Union
import fire
import torch
from tqdm import tqdm
from transformers import LlamaForCausalLM  # @manual
def insert_chunk(name: str, tensor: torch.Tensor, dim: int):
    tensors = tensor.chunk(num_shards, dim=dim)
    for i, tensor in enumerate(tensors):
        state_dict[i][name] = tensor.clone()