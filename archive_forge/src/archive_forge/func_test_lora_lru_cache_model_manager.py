import os
from typing import List
import pytest
import torch
from safetensors.torch import load_file
from torch import nn
from vllm.config import LoRAConfig
from vllm.lora.layers import (ColumnParallelLinearWithLoRA,
from vllm.lora.lora import LoRALayerWeights, PackedLoRALayerWeights
from vllm.lora.models import (LoRAModel, LoRAModelManager,
from vllm.lora.request import LoRARequest
from vllm.lora.worker_manager import (LRUCacheWorkerLoRAManager,
from vllm.model_executor.layers.linear import RowParallelLinear
def test_lora_lru_cache_model_manager(dist_init, dummy_model):
    model = dummy_model
    model.supported_lora_modules = ['dense1', 'dense2', 'lm_head']
    model.packed_modules_mapping = {}
    model_lora1 = create_lora(1, model, ['layer1.dense1', 'dense2', 'lm_head'])
    model_lora2 = create_lora(2, model, ['dense1', 'dense2', 'lm_head'])
    model_lora3 = create_lora(3, model, ['dense1', 'dense2', 'lm_head'])
    manager = LRUCacheLoRAModelManager(model, 2, 2, 2, LoRAConfig(max_lora_rank=8, max_cpu_loras=3, max_loras=2))
    assert all((x is None for x in manager.lora_index_to_id))
    assert manager.add_lora(model_lora1)
    assert manager.activate_lora(1)
    assert manager.lora_index_to_id[0] == 1
    assert not manager.add_lora(model_lora1)
    assert not manager.activate_lora(1)
    assert manager.add_lora(model_lora2)
    assert manager.activate_lora(2)
    assert manager.lora_index_to_id[0] == 1
    assert manager.lora_index_to_id[1] == 2
    assert not manager.add_lora(model_lora2)
    assert not manager.activate_lora(2)
    assert manager.add_lora(model_lora3)
    assert manager.lora_index_to_id[0] == 1
    assert manager.lora_index_to_id[1] == 2
    assert manager.activate_lora(3)
    assert manager.lora_index_to_id[0] == 3
    assert manager.lora_index_to_id[1] == 2
    assert manager.remove_lora(model_lora2.id)
    assert manager.lora_index_to_id[1] is None
    assert not manager.remove_lora(model_lora2.id)
    assert manager.remove_lora(model_lora1.id)
    assert not manager.remove_lora(model_lora1.id)
    assert manager.add_lora(model_lora1)
    assert manager.activate_lora(1)
    assert manager.lora_index_to_id[0] == 3
    assert manager.lora_index_to_id[1] == 1
    assert manager.add_lora(model_lora2)
    assert manager.deactivate_lora(3)
    assert manager.lora_index_to_id[0] is None
    assert manager.lora_index_to_id[1] == 1
    assert manager.activate_lora(2)
    assert manager.lora_index_to_id[0] == 2
    assert manager.lora_index_to_id[1] == 1
    assert manager.activate_lora(3)
    assert manager.lora_index_to_id[0] == 2
    assert manager.lora_index_to_id[1] == 3