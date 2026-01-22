import functools
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from torch.distributed.fsdp.wrap import (
def get_size_policy(min_params=100000000.0):
    num_wrap_policy = functools.partial(size_based_auto_wrap_policy, min_num_params=min_params)
    return num_wrap_policy