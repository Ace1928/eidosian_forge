import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PretrainedConfig
from vllm.config import LoRAConfig
from vllm.lora.punica import add_lora, add_lora_slice, bgmv
from vllm.model_executor.layers.sampler import Sampler
from vllm.model_executor.parallel_utils.communication_op import (
from vllm.model_executor.layers.linear import (ColumnParallelLinear,
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding, ParallelLMHead
from vllm.model_executor.parallel_utils.parallel_state import (
from vllm.model_executor.parallel_utils.utils import split_tensor_along_last_dim
def from_layer_sampler(layer: Sampler, lm_head: ParallelLMHead, max_loras: int, lora_config: LoRAConfig, model_config: Optional[PretrainedConfig]=None) -> SamplerWithLoRA:
    ret = SamplerWithLoRA(layer, lm_head.embedding_dim, lm_head.weight.dtype, lm_head.weight.device)
    ret.create_lora_weights(max_loras, lora_config, model_config)
    return ret