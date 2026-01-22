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
def test_from_lora_tensors(sql_lora_files):
    tensors = load_file(os.path.join(sql_lora_files, 'adapter_model.safetensors'))
    new_embeddings = load_file(os.path.join(sql_lora_files, 'new_embeddings.safetensors'))
    lora_model = LoRAModel.from_lora_tensors(1, 8, 16, tensors, 'cuda', embeddings=new_embeddings, embedding_modules=EMBEDDING_MODULES, embedding_padding_modules=EMBEDDING_PADDING_MODULES)
    for module_name, lora in lora_model.loras.items():
        assert lora.module_name == module_name
        assert lora.rank == 8
        assert lora.lora_alpha == 16
        assert lora.lora_a is not None
        assert lora.lora_b is not None
        assert lora.lora_a.shape[1] == lora.lora_b.shape[0], f'lora.lora_a.shape={lora.lora_a.shape!r}, lora.lora_b.shape={lora.lora_b.shape!r}'
        assert lora.lora_a.shape[1] == 8
        embeddings_module = next((k for k in EMBEDDING_MODULES if k in module_name), None)
        if embeddings_module:
            assert torch.equal(lora.embeddings_tensor, new_embeddings[EMBEDDING_MODULES[embeddings_module]].to(device=lora.embeddings_tensor.device))
        else:
            assert lora.embeddings_tensor is None