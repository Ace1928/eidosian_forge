from __future__ import annotations
import warnings
from typing import Any, Optional, Union
from torch import nn
from tqdm import tqdm
from peft.tuners import adalora, loha, lokr, lora, oft
from peft.tuners.tuners_utils import BaseTuner, BaseTunerLayer, check_target_module_exists
from peft.utils import (
def merge_recursively(module):
    path = []
    layer = module
    while hasattr(layer, 'base_layer'):
        path.append(layer)
        layer = layer.base_layer
    for layer_before, layer_after in zip(path[:-1], path[1:]):
        layer_after.merge(safe_merge=safe_merge, adapter_names=adapter_names)
        layer_before.base_layer = layer_after.base_layer
    module.merge(safe_merge=safe_merge, adapter_names=adapter_names)