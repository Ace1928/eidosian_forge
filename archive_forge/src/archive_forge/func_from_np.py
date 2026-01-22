from collections import OrderedDict
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Set, Tuple, Union, cast
import numpy as np
import torch
from torch.nn.utils.rnn import PackedSequence
def from_np(ndarray_or_container: Union[np.ndarray, Dict, List, Tuple, Set]) -> Any:
    """Convert a ndarray or a container to tensor."""
    return apply_to_type(lambda x: isinstance(x, np.ndarray), lambda x: torch.from_numpy(x), ndarray_or_container)