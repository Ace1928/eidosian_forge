import copy
import json
import logging
import math
import os
import re
from typing import (Any, Callable, Dict, Hashable, List, Optional, Tuple, Type)
import safetensors.torch
import torch
from torch import nn
from vllm.config import LoRAConfig
from vllm.utils import LRUCache, in_wsl
from vllm.lora.layers import BaseLayerWithLoRA, LoRAMapping, from_layer, from_layer_sampler
from vllm.lora.lora import LoRALayerWeights, PackedLoRALayerWeights
from vllm.lora.utils import parse_fine_tuned_lora_name, replace_submodule
def create_dummy_lora(self, lora_id: int, rank: int, embedding_modules: Optional[Dict[str, str]]=None) -> LoRAModel:
    """Create zero-initialized LoRAModel for warmup."""
    model = LoRAModel(lora_id, rank, {})
    for module_name, module in self.model.named_modules():
        if not self._match_target_modules(module_name) or not isinstance(module, BaseLayerWithLoRA):
            continue
        parts = module_name.split('.')
        if module_name not in self.packed_modules:
            if parts[-1] in embedding_modules:
                input_dim = module.base_layer.org_vocab_size + self.lora_config.lora_extra_vocab_size if hasattr(module.base_layer, 'org_vocab_size') else module.base_layer.weight.shape[1]
                output_dim = module.base_layer.embedding_dim if hasattr(module.base_layer, 'embedding_dim') else module.base_layer.weight.shape[0]
                embeddings_tensor_dim = module.base_layer.embedding_dim if hasattr(module.base_layer, 'embedding_dim') else module.base_layer.weight.shape[1]
                lora = LoRALayerWeights.create_dummy_lora_weights(module_name, input_dim, output_dim, rank, module.lora_a_stacked.dtype, 'cpu', embeddings_tensor_dim=embeddings_tensor_dim)
            else:
                lora = LoRALayerWeights.create_dummy_lora_weights(module_name, module.lora_a_stacked.shape[-1], module.lora_b_stacked.shape[-2], rank, module.lora_a_stacked.dtype, 'cpu')
            lora.optimize()
        else:
            parts = module_name.split('.')
            replacements = self.packed_modules_mapping[parts[-1]]
            subloras = []
            for i, r in enumerate(replacements):
                lora = LoRALayerWeights.create_dummy_lora_weights(module_name + '.' + r, module.lora_a_stacked[i].shape[-1], module.lora_b_stacked[i].shape[-2], rank, module.lora_a_stacked[i].dtype, 'cpu')
                lora.optimize()
                subloras.append(lora)
            lora = PackedLoRALayerWeights.pack(subloras)
        model.loras[module_name] = lora
    return model