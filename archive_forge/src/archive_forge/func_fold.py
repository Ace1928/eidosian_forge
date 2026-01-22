import os
from dataclasses import replace
from itertools import zip_longest
from typing import Any, List, Optional, Set, Tuple, Union
import torch
from ..common import _get_storage_base, get_operator, register_operator
from .attn_bias import (
from .common import (
def fold(x):
    if x.stride(3) == 0:
        return x[:, :, :, 0]
    return x.reshape([x.shape[0], x.shape[1], -1, x.shape[4]])