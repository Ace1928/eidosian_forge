import re
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from ...processing_utils import ProcessorMixin
from ...tokenization_utils_base import PaddingStrategy, TruncationStrategy
from ...utils import TensorType, is_torch_available, logging, requires_backends
def full_unpacked_stream_to_tensor(all_bi_tokens_to_place: List[int], full_unpacked_stream: List['torch.Tensor'], fill_value: int, batch_size: int, new_seq_len: int, offset: int) -> 'torch.Tensor':
    """Takes an unpacked stream of tokens (i.e. a list of tensors, one for each item in the batch) and does
    the required padding to create a single tensor for the batch of shape batch_size x new_seq_len.
    """
    assert len(all_bi_tokens_to_place) == batch_size
    assert len(full_unpacked_stream) == batch_size
    new_padded_tensor = torch.full([batch_size, new_seq_len], fill_value=fill_value, dtype=full_unpacked_stream[0].dtype, device=full_unpacked_stream[0].device)
    for bi in range(batch_size):
        tokens_to_place = all_bi_tokens_to_place[bi]
        new_padded_tensor[bi, :tokens_to_place] = full_unpacked_stream[bi][offset:tokens_to_place + offset]
    return new_padded_tensor