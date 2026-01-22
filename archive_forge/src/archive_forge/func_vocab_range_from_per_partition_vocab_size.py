from typing import Optional, Sequence
import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from vllm.model_executor.parallel_utils.parallel_state import (
from vllm.model_executor.parallel_utils.utils import divide
from vllm.model_executor.parallel_utils.communication_op import (
from vllm.model_executor.utils import set_weight_attrs
def vocab_range_from_per_partition_vocab_size(per_partition_vocab_size: int, rank: int) -> Sequence[int]:
    index_f = rank * per_partition_vocab_size
    index_l = index_f + per_partition_vocab_size
    return (index_f, index_l)