import os
import re
from enum import Enum
from inspect import signature
from typing import Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
from packaging import version
from transformers.utils import logging
import onnxruntime as ort
from ..exporters.onnx import OnnxConfig, OnnxConfigWithLoss
from ..utils.import_utils import _is_package_available
def parse_device(device: Union[torch.device, str, int]) -> Tuple[torch.device, Dict]:
    """Gets the relevant torch.device from the passed device, and if relevant the provider options (e.g. to set the GPU id)."""
    if device == -1:
        device = torch.device('cpu')
    else:
        device = torch._C._nn._parse_to(device)[0]
    provider_options = {}
    if device.type == 'cuda':
        if device.index is None:
            device = torch.device('cuda:0')
        provider_options['device_id'] = device.index
    return (device, provider_options)