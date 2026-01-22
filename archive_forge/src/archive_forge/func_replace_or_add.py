import copy
import math
import warnings
import zlib
from typing import Callable, Iterator, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from ...generation.configuration_utils import GenerationConfig
from ...generation.logits_process import (
from ...generation.stopping_criteria import StoppingCriteriaList
from ...modeling_outputs import BaseModelOutput
from ...utils import logging
from .tokenization_whisper import TASK_IDS, TO_LANGUAGE_CODE
def replace_or_add(lst: List[int], num: int, itr: Iterator[int]):
    """short function to replace num with a itr in lst"""
    found = any((i in lst for i in itr))
    if found:
        lst = [num if i in itr else i for i in lst]
    else:
        lst.append(num)
    return lst