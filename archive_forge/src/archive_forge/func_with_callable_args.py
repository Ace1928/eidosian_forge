import re
import warnings
from abc import ABCMeta, abstractmethod
from collections import OrderedDict
from functools import partial
from typing import Any, List, Tuple, Optional, Dict
import torch
import torch.nn as nn
from torch.ao.quantization.utils import (
def with_callable_args(self, **kwargs):
    result = _PartialWrapper(p=self.p)
    result.callable_args = {**self.callable_args, **kwargs}
    return result