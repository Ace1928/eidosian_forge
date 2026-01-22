import math
import re
from typing import TYPE_CHECKING, Dict
from ._base import MAX_SHARD_SIZE, StateDictSplit, split_state_dict_into_shards_factory
def get_tensor_size(tensor: 'tf.Tensor') -> int:
    return math.ceil(tensor.numpy().size * _dtype_byte_size_tf(tensor.dtype))