from dataclasses import dataclass
from functools import reduce  # Required in Python 3
import operator
from typing import Callable, Optional, Tuple
import warnings
from warnings import warn
import torch
import bitsandbytes.functional as F
def supports_igemmlt(device: torch.device) -> bool:
    """check if this device supports the optimized int8 kernel"""
    if torch.cuda.get_device_capability(device=device) < (7, 5):
        return False
    device_name = torch.cuda.get_device_name(device=device)
    nvidia16_models = ('GTX 1630', 'GTX 1650', 'GTX 1660')
    if any((model_name in device_name for model_name in nvidia16_models)):
        return False
    return True