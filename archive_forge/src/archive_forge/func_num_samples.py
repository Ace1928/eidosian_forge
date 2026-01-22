import torch
from torch import Tensor
from typing import Iterator, Iterable, Optional, Sequence, List, TypeVar, Generic, Sized, Union
@property
def num_samples(self) -> int:
    if self._num_samples is None:
        return len(self.data_source)
    return self._num_samples