import contextlib
import time
from typing import Dict, List, Optional, Tuple, Set, Union
import numpy as np
import torch
import torch.nn as nn
from vllm.config import (DeviceConfig, ModelConfig, LoRAConfig, ParallelConfig,
from vllm.logger import init_logger
from vllm.model_executor import get_model, InputMetadata, SamplingMetadata
from vllm.model_executor.parallel_utils import cupy_utils
from vllm.model_executor.parallel_utils.communication_op import (
from vllm.model_executor.parallel_utils.parallel_state import (
from vllm.model_executor.parallel_utils import custom_all_reduce
from vllm.sampling_params import SamplingParams, SamplingType
from vllm.sequence import SamplerOutput, SequenceData, SequenceGroupMetadata
from vllm.lora.worker_manager import LRUCacheWorkerLoRAManager
from vllm.lora.layers import LoRAMapping
from vllm.lora.request import LoRARequest
from vllm.utils import in_wsl
@torch.inference_mode()
def profile_run(self) -> None:
    vocab_size = self.model_config.get_vocab_size()
    sampling_params = SamplingParams(top_p=0.99, top_k=vocab_size - 1)
    max_num_batched_tokens = self.scheduler_config.max_num_batched_tokens
    max_num_seqs = self.scheduler_config.max_num_seqs
    dummy_lora_requests = []
    dummy_lora_requests_per_seq = []
    if self.lora_config:
        for idx in range(self.lora_config.max_loras):
            lora_id = idx + 1
            dummy_lora_request = LoRARequest(lora_name=f'warmup_{lora_id}', lora_int_id=lora_id, lora_local_path='/not/a/real/path')
            self.lora_manager.add_dummy_lora(dummy_lora_request, rank=LORA_WARMUP_RANK)
            dummy_lora_requests.append(dummy_lora_request)
        dummy_lora_requests_per_seq = [dummy_lora_requests[idx % len(dummy_lora_requests)] for idx in range(max_num_seqs)]
    seqs: List[SequenceGroupMetadata] = []
    for group_id in range(max_num_seqs):
        seq_len = max_num_batched_tokens // max_num_seqs + (group_id < max_num_batched_tokens % max_num_seqs)
        seq_data = SequenceData([0] * seq_len)
        seq = SequenceGroupMetadata(request_id=str(group_id), is_prompt=True, seq_data={group_id: seq_data}, sampling_params=sampling_params, block_tables=None, lora_request=dummy_lora_requests_per_seq[group_id] if dummy_lora_requests_per_seq else None)
        seqs.append(seq)
    num_layers = self.model_config.get_num_layers(self.parallel_config)
    kv_caches = [(None, None)] * num_layers
    self.execute_model(seqs, kv_caches)
    torch.cuda.synchronize()
    return