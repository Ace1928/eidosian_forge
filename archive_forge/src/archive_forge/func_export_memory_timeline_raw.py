import collections
import dataclasses
import enum
import itertools as it
import logging
from typing import (
from typing_extensions import Literal
import torch
from torch._C import FunctionSchema
from torch._C._autograd import _ProfilerResult
from torch._C._profiler import (
from torch._utils import _element_size
from torch.profiler import _utils
def export_memory_timeline_raw(self, path, device_str) -> None:
    """Saves the memory timeline as raw memory event tuples in the
        form of (timestamp, action, numbytes, category)
        as a JSON formatted file to the given path for the given
        device."""
    device = torch.device(device_str)
    raw_events: List[Tuple[int, int, int, int]] = []

    def get_category_index(key, version):
        category = self.categories.get(key, version) if isinstance(key, TensorKey) else None
        return _CATEGORY_TO_INDEX[category]
    for t, action, (key, version), numbytes in self.timeline:
        if key.device != device:
            continue
        if action in (Action.PREEXISTING, Action.CREATE):
            raw_events.append((t, _ACTION_TO_INDEX[action], numbytes, get_category_index(key, version)))
        elif action == Action.INCREMENT_VERSION:
            raw_events.append((t, _ACTION_TO_INDEX[action], -numbytes, get_category_index(key, version)))
            raw_events.append((t, _ACTION_TO_INDEX[action], numbytes, get_category_index(key, version + 1)))
        elif action == Action.DESTROY:
            raw_events.append((t, _ACTION_TO_INDEX[action], -numbytes, get_category_index(key, version)))
        else:
            raise ValueError(f'Unknown action: {action}')
    import json
    with open(path, 'w') as f:
        json.dump(raw_events, f)