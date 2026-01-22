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
def get_device_for_provider(provider: str, provider_options: Dict) -> torch.device:
    """
    Gets the PyTorch device (CPU/CUDA) associated with an ONNX Runtime provider.
    """
    if provider in ['CUDAExecutionProvider', 'TensorrtExecutionProvider', 'ROCMExecutionProvider']:
        return torch.device(f'cuda:{provider_options['device_id']}')
    else:
        return torch.device('cpu')