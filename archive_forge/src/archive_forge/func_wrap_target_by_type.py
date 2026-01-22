from __future__ import annotations
import collections.abc
import contextlib
from collections import defaultdict
from copy import copy
import torch
from torchvision import datasets, tv_tensors
from torchvision.transforms.v2 import functional as F
def wrap_target_by_type(target, *, target_types, type_wrappers):
    if not isinstance(target, (tuple, list)):
        target = [target]
    wrapped_target = tuple((type_wrappers.get(target_type, identity)(item) for target_type, item in zip(target_types, target)))
    if len(wrapped_target) == 1:
        wrapped_target = wrapped_target[0]
    return wrapped_target