import gc
import time
from collections import namedtuple
from dataclasses import dataclass, field
from functools import partial
from typing import Callable, Optional, Sequence, Union
import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import Tensor
from torch.profiler import ProfilerActivity, profile, record_function
from transformers.generation import GreedySearchDecoderOnlyOutput, SampleDecoderOnlyOutput
@torch.inference_mode()
def update_graph_cache(model, cache, batch_size, seqlen_og, max_seqlen, decoding_seqlens=(1,), tensor_parallel=1, dtype=None, n_warmups=2):
    if cache is None:
        cache = DecodingCGCache()
    param_example = next(iter(model.parameters()))
    device = param_example.device
    if dtype is None:
        dtype = param_example.dtype
    if (device, dtype) != (cache.device, cache.dtype) or batch_size > cache.max_batch_size or max_seqlen > cache.max_seqlen:
        cache.callables = {}
        cache.mempool = None
        cache.inference_params = None
        gc.collect()
        cache.device, cache.dtype = (device, dtype)
        cache.max_batch_size, cache.max_seqlen = (batch_size, max_seqlen)
        if hasattr(model, 'allocate_inference_cache'):
            inf_cache = model.allocate_inference_cache(batch_size, max_seqlen, dtype)
        else:
            headdim = getattr(model.config, 'head_dim', model.config.hidden_size // model.config.num_attention_heads)
            inf_cache = allocate_inference_cache(batch_size, max_seqlen, model.config.num_attention_heads // tensor_parallel, headdim, model.config.num_hidden_layers, device, dtype)
        lengths_per_sample = torch.full((batch_size,), seqlen_og, dtype=torch.int32, device=device)
        cache.inference_params = InferenceParams(max_seqlen=max_seqlen, max_batch_size=batch_size, seqlen_offset=seqlen_og, key_value_memory_dict=inf_cache, lengths_per_sample=lengths_per_sample)
        cache.mempool = torch.cuda.graphs.graph_pool_handle()
    for decoding_seqlen in decoding_seqlens:
        if (batch_size, decoding_seqlen) not in cache.callables:
            cache.callables[batch_size, decoding_seqlen] = capture_graph(model, cache.inference_params, batch_size, max_seqlen, decoding_seqlen=decoding_seqlen, mempool=cache.mempool, n_warmups=n_warmups)

    def dispatch(input_ids, position_ids, seqlen):
        batch_size, decoding_seqlen = input_ids.shape[:2]
        return cache.callables[batch_size, decoding_seqlen](input_ids, position_ids, seqlen)
    cache.run = dispatch
    cache.inference_params.seqlen_offset = 0
    return cache