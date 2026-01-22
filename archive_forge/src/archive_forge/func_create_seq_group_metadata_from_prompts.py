import torch
from typing import List, Optional, Dict
from vllm.worker.worker import Worker
from vllm.utils import get_distributed_init_method, get_ip, get_open_port
from vllm.engine.arg_utils import EngineArgs
from vllm.sequence import SequenceGroupMetadata, SequenceData
from vllm.sampling_params import SamplingParams
from vllm.worker.cache_engine import CacheEngine
from vllm.model_executor.utils import set_random_seed
from dataclasses import dataclass, fields
def create_seq_group_metadata_from_prompts(prompts: List[List[int]], num_gpu_blocks: int, block_size: int, final_seq_lens: List[int], continuations: Optional[List[List[int]]]=None, num_tokens_processed: Optional[List[int]]=None, seq_ids: Optional[List[int]]=None) -> List[SequenceGroupMetadata]:
    if continuations is None:
        continuations = [[] for _ in prompts]
    if num_tokens_processed is None:
        num_tokens_processed = []
        for continuation, prompt in zip(continuations, prompts):
            if not continuation:
                num_tokens_processed.append(0)
            else:
                num_tokens_processed.append(len(continuation) + len(prompt) - 1)
    if seq_ids is None:
        seq_ids = list((i for i, _ in enumerate(prompts)))
    free_gpu_blocks = list(range(num_gpu_blocks))
    block_allocations = {i: [free_gpu_blocks.pop() for _ in range(round_up_to_next_block(final_len, block_size))] for i, final_len in enumerate(final_seq_lens)}
    return [SequenceGroupMetadata(request_id=str(i), is_prompt=len(cont_token_ids) == 0, seq_data={i: SequenceData(prompt_token_ids=prompt_token_ids[:] + cont_token_ids[:])}, sampling_params=SamplingParams(temperature=0.0), block_tables={i: block_allocations[i][:]}) for i, (prompt_token_ids, cont_token_ids, num_tokens_saved) in enumerate(zip(prompts, continuations, num_tokens_processed))]