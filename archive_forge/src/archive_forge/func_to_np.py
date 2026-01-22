from collections import OrderedDict
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Set, Tuple, Union, cast
import numpy as np
import torch
from torch.nn.utils.rnn import PackedSequence
def to_np(tensor_or_container: Union[torch.Tensor, Dict, List, Tuple, Set]) -> Any:
    """Convert a tensor or a container to numpy."""
    return apply_to_type(torch.is_tensor, lambda x: x.cpu().numpy(), tensor_or_container)