import copy
from enum import IntEnum
from functools import cached_property
from typing import Callable, List, Optional, Union
import torch
Deep copy excluding LogitsProcessor objects.

        LogitsProcessor objects are excluded because they may contain an
        arbitrary, nontrivial amount of data.
        See https://github.com/vllm-project/vllm/issues/3087
        