import pytest
import random
from copy import deepcopy
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple
import torch
import torch.nn.functional as F
from vllm.lora.layers import (
from vllm.lora.models import LoRALayerWeights, convert_mapping, PackedLoRALayerWeights
from vllm.config import LoRAConfig
from vllm.model_executor.layers.sampler import Sampler
from vllm.model_executor.layers.linear import (ColumnParallelLinear,
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding, ParallelLMHead
from vllm.model_executor.utils import set_random_seed
from .utils import DummyLoRAManager
@torch.inference_mode()
@pytest.mark.parametrize('num_loras', [1, 2, 4, 8])
@pytest.mark.parametrize('device', CUDA_DEVICES)
def test_lm_head_sampler(dist_init, num_loras, device) -> None:
    torch.set_default_device(device)
    max_loras = 8
    lora_config = LoRAConfig(max_loras=max_loras, max_lora_rank=8, lora_dtype=torch.float16)

    def create_random_sampler_layer():
        linear = ParallelLMHead(32000 + lora_config.lora_extra_vocab_size, 1024, 32000)
        linear.weight.data = torch.rand_like(linear.weight.data)
        linear.weight.data[:, 32000:] = 0
        sampler = Sampler(32000 + lora_config.lora_extra_vocab_size, 32000)
        lora_sampler = SamplerWithLoRA(sampler, 1024, linear.weight.dtype, linear.weight.device)
        lora_sampler.create_lora_weights(max_loras, lora_config)
        return (linear, sampler, lora_sampler)
    for i in range(10):
        set_random_seed(i)
        id_to_index = get_random_id_to_index(num_loras, max_loras)
        linear, sampler, lora_sampler = create_random_sampler_layer()
        lora_dict, _ = populate_loras(id_to_index, layer=lora_sampler, layer_weights=linear.weight, generate_embeddings_tensor=1024)
        embeddings_tensor = list(lora_dict.values())[0].embeddings_tensor
        embeddings_tensor_len = embeddings_tensor.shape[0]
        inputs, index_mapping, prompt_mapping = create_random_inputs(active_lora_ids=list(lora_dict.keys()), num_inputs=8 * num_loras, input_size=(1, 1024), input_range=(0, 1), input_type=torch.float32)
        lora_mapping = LoRAMapping(index_mapping, prompt_mapping)
        input_ = torch.rand(20, 1024)
        mapping_info = convert_mapping(lora_mapping, id_to_index, max_loras, 32000, lora_config.lora_extra_vocab_size)
        lora_sampler.set_mapping(*mapping_info)
        lora_result = lora_sampler._get_logits(hidden_states=torch.cat(inputs), embedding=linear.weight, embedding_bias=None)
        original_weight = linear.weight.clone()
        linear.weight[sampler.org_vocab_size:sampler.org_vocab_size + embeddings_tensor_len] = embeddings_tensor
        sampler.org_vocab_size = 32000 + lora_config.lora_extra_vocab_size
        expected_results = []
        for input_, lora_id in zip(inputs, prompt_mapping):
            lora = lora_dict[lora_id]
            result = sampler._get_logits(hidden_states=input_, embedding=linear.weight, embedding_bias=None)
            result[:, 32000 + embeddings_tensor_len:] = float('-inf')
            result += input_ @ lora.lora_a @ lora.lora_b * lora.scaling
            expected_results.append(result)
        expected_result = torch.cat(expected_results)
        sampler.org_vocab_size = 32000
        for slot_idx in range(max_loras):
            lora_sampler.reset_lora(slot_idx)
        inputs, index_mapping, prompt_mapping = create_random_inputs(active_lora_ids=[0], num_inputs=8 * num_loras * 3, input_size=(1, 1024), input_range=(0, 1), input_type=torch.float32)
        lora_mapping = LoRAMapping(index_mapping, prompt_mapping)
        mapping_info = convert_mapping(lora_mapping, id_to_index, max_loras, 32000, lora_config.lora_extra_vocab_size)
        lora_sampler.set_mapping(*mapping_info)
        lora_result = lora_sampler._get_logits(hidden_states=torch.cat(inputs), embedding=original_weight, embedding_bias=None)[:, :32000]
        expected_result = sampler._get_logits(hidden_states=torch.cat(inputs), embedding=original_weight, embedding_bias=None)
        rtol, atol = TOLERANCES[lora_result.dtype]
        assert torch.allclose(lora_result, expected_result, rtol=rtol, atol=atol)