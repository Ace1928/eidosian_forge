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
def prepare_input_tensors(self, seq_group_metadata_list: Optional[List[SequenceGroupMetadata]]) -> Tuple[torch.Tensor, torch.Tensor, InputMetadata, SamplingMetadata, Set[int], LoRAMapping]:
    if self.is_driver_worker:
        is_prompt = seq_group_metadata_list[0].is_prompt
        if is_prompt:
            input_tokens, input_positions, input_metadata, prompt_lens, subquery_lens, lora_index_mapping, lora_prompt_mapping, lora_requests = self._prepare_prompt(seq_group_metadata_list)
        else:
            input_tokens, input_positions, input_metadata, lora_index_mapping, lora_prompt_mapping, lora_requests = self._prepare_decode(seq_group_metadata_list)
            prompt_lens = []
            subquery_lens = None
        sampling_metadata = self._prepare_sample(seq_group_metadata_list, prompt_lens, subquery_lens)
        if self.lora_config:
            flat_lora_index_mapping = [item for sublist in lora_index_mapping for item in sublist]
            lora_mapping = LoRAMapping(flat_lora_index_mapping, lora_prompt_mapping)
        else:
            lora_mapping = None
        metadata_dict = {'input_tokens': input_tokens, 'input_positions': input_positions, 'is_prompt': input_metadata.is_prompt, 'slot_mapping': input_metadata.slot_mapping, 'prompt_lens': input_metadata.prompt_lens, 'max_seq_len': input_metadata.max_seq_len, 'start_loc': input_metadata.start_loc, 'max_context_len': input_metadata.max_context_len, 'context_lens': input_metadata.context_lens, 'block_tables': input_metadata.block_tables, 'use_cuda_graph': input_metadata.use_cuda_graph, 'kv_cache_dtype': input_metadata.kv_cache_dtype, 'selected_token_indices': sampling_metadata.selected_token_indices, 'lora_requests': lora_requests, 'lora_mapping': lora_mapping}
        broadcast_tensor_dict(metadata_dict, src=0)
    else:
        metadata_dict = broadcast_tensor_dict(src=0)
        input_tokens = metadata_dict['input_tokens']
        input_positions = metadata_dict['input_positions']
        lora_mapping = metadata_dict['lora_mapping']
        lora_requests = metadata_dict['lora_requests']
        input_metadata = InputMetadata(is_prompt=metadata_dict['is_prompt'], slot_mapping=metadata_dict['slot_mapping'], prompt_lens=metadata_dict['prompt_lens'], max_seq_len=metadata_dict['max_seq_len'], start_loc=metadata_dict['start_loc'], max_context_len=metadata_dict['max_context_len'], context_lens=metadata_dict['context_lens'], block_tables=metadata_dict['block_tables'], use_cuda_graph=metadata_dict['use_cuda_graph'], kv_cache_dtype=metadata_dict['kv_cache_dtype'])
        sampling_metadata = SamplingMetadata(seq_groups=None, seq_data=None, prompt_lens=None, selected_token_indices=metadata_dict['selected_token_indices'], categorized_sample_indices=None, generators=None, perform_sampling=False)
    return (input_tokens, input_positions, input_metadata, sampling_metadata, lora_requests, lora_mapping)