import copy
from typing import (
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.utils._named_member_accessor import NamedMemberAccessor
def transpose_stack(tuple_of_tuple_of_tensors: Tuple[Tuple[Tensor, ...], ...]) -> Tuple[Tensor, ...]:
    tuple_of_tuple_of_tensors = tuple(zip(*tuple_of_tuple_of_tensors))
    results = tuple((torch.stack(shards).detach() for shards in tuple_of_tuple_of_tensors))
    return results