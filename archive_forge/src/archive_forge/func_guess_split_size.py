import os
from typing import Union, Optional, Tuple, Any, List, Sized, TypeVar
import itertools
from collections import namedtuple
import parlai.utils.logging as logging
import torch.optim
@staticmethod
def guess_split_size(item: Chunk, num_gpus: Optional[int]=None, dim=0) -> int:
    """
        Estimate the number of chunks we should split the batch into via heuristics.
        """
    if num_gpus is None:
        num_gpus = torch.cuda.device_count()
    if isinstance(item, torch.Tensor):
        if num_gpus == 1:
            return item.size(dim)
        return max(1, item.size(dim) // int(num_gpus * 2))
    elif isinstance(item, tuple):
        return PipelineHelper.guess_split_size(item[0], num_gpus)
    elif isinstance(item, dict):
        return PipelineHelper.guess_split_size(list(item.values())[0], num_gpus)
    raise TypeError(f'Cannot determine split size for {type(item)}')