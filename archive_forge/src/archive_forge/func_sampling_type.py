import copy
from enum import IntEnum
from functools import cached_property
from typing import Callable, List, Optional, Union
import torch
@cached_property
def sampling_type(self) -> SamplingType:
    if self.use_beam_search:
        return SamplingType.BEAM
    if self.temperature < _SAMPLING_EPS:
        return SamplingType.GREEDY
    if self.seed is not None:
        return SamplingType.RANDOM_SEED
    return SamplingType.RANDOM