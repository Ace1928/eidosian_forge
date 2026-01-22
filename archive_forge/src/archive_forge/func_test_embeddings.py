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
def test_embeddings(dist_init, num_loras, device) -> None:
    torch.set_default_device(device)
    max_loras = 8
    lora_config = LoRAConfig(max_loras=max_loras, max_lora_rank=8, lora_dtype=torch.float16)

    def create_random_embedding_layer():
        embedding = VocabParallelEmbedding(512, 256)
        embedding.weight.data = torch.rand_like(embedding.weight.data)
        embedding.weight.data[512:, :] = 0
        lora_embedding = VocabParallelEmbeddingWithLoRA(embedding)
        lora_embedding.create_lora_weights(max_loras, lora_config)
        return (embedding, lora_embedding)
    for i in range(10):
        set_random_seed(i)
        id_to_index = get_random_id_to_index(num_loras, max_loras)
        embedding, lora_embedding = create_random_embedding_layer()
        lora_dict, _ = populate_loras(id_to_index, layer=lora_embedding, layer_weights=embedding.weight.T)
        inputs, index_mapping, prompt_mapping = create_random_inputs(active_lora_ids=list(lora_dict.keys()), num_inputs=num_loras * 3, input_size=(200,), input_range=(1, 512))
        lora_mapping = LoRAMapping(index_mapping, prompt_mapping)
        mapping_info = convert_mapping(lora_mapping, id_to_index, max_loras, 512, lora_config.lora_extra_vocab_size)
        lora_embedding.set_mapping(*mapping_info)
        lora_result = lora_embedding(torch.cat(inputs))
        expected_results = []
        for input_, lora_id in zip(inputs, prompt_mapping):
            lora = lora_dict[lora_id]
            result = embedding(input_)
            after_a = F.embedding(input_, lora.lora_a)
            result += after_a @ lora.lora_b
            expected_results.append(result)
        expected_result = torch.cat(expected_results)
        rtol, atol = TOLERANCES[lora_result.dtype]
        assert torch.allclose(lora_result, expected_result, rtol=rtol, atol=atol)
        for slot_idx in range(max_loras):
            lora_embedding.reset_lora(slot_idx)
        inputs, index_mapping, prompt_mapping = create_random_inputs(active_lora_ids=[0], num_inputs=num_loras * 3, input_size=(200,), input_range=(1, 512))
        lora_mapping = LoRAMapping(index_mapping, prompt_mapping)
        mapping_info = convert_mapping(lora_mapping, id_to_index, max_loras, 512, lora_config.lora_extra_vocab_size)
        lora_embedding.set_mapping(*mapping_info)
        lora_result = lora_embedding(torch.cat(inputs))
        expected_result = embedding(torch.cat(inputs))
        rtol, atol = TOLERANCES[lora_result.dtype]
        assert torch.allclose(lora_result, expected_result, rtol=rtol, atol=atol)