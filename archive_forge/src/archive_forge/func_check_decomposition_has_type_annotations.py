import torch
from torch import Tensor
import inspect
import warnings
from typing import Dict, List, Optional, Set
from torch.types import Number
def check_decomposition_has_type_annotations(f):
    inspect_empty = inspect._empty
    sig = inspect.signature(f)
    for param in sig.parameters.values():
        assert param.annotation != inspect_empty, f'No signature on param {param.name} for function {f.name}'
    assert sig.return_annotation != inspect_empty, f'No return annotation for function {f.name}'