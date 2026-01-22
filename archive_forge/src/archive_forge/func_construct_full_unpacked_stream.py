import re
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from ...processing_utils import ProcessorMixin
from ...tokenization_utils_base import PaddingStrategy, TruncationStrategy
from ...utils import TensorType, is_torch_available, logging, requires_backends
def construct_full_unpacked_stream(num_real_text_tokens: Union[List[List[int]], 'torch.Tensor'], input_stream: 'torch.Tensor', image_tokens: List[List['torch.Tensor']], batch_size: int, num_sub_sequences: int) -> List['torch.Tensor']:
    """Takes an input_stream tensor of shape B x S x ?. For each subsequence, adds any required
    padding to account for images and then unpacks the subsequences to create a single sequence per item in the batch.
    Returns a list of tensors, one for each item in the batch."""
    all_bi_stream = []
    for batch_index in range(batch_size):
        all_si_stream = []
        image_adjustment = image_tokens[batch_index][0]
        subsequence_stream = torch.cat([image_adjustment, input_stream[batch_index, 0]], dim=0)
        num_real_tokens = image_adjustment.shape[0] + num_real_text_tokens[batch_index][0]
        all_si_stream.append(subsequence_stream[:num_real_tokens])
        all_bi_stream.append(torch.cat(all_si_stream, dim=0))
    return all_bi_stream