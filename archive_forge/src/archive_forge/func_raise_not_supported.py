from __future__ import annotations
import collections.abc
import contextlib
from collections import defaultdict
from copy import copy
import torch
from torchvision import datasets, tv_tensors
from torchvision.transforms.v2 import functional as F
def raise_not_supported(description):
    raise RuntimeError(f'{description} is currently not supported by this wrapper. If this would be helpful for you, please open an issue at https://github.com/pytorch/vision/issues.')