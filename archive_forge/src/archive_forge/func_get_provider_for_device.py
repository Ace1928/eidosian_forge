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
def get_provider_for_device(device: torch.device) -> str:
    """
    Gets the ONNX Runtime provider associated with the PyTorch device (CPU/CUDA).
    """
    if device.type.lower() == 'cuda':
        if 'ROCMExecutionProvider' in ort.get_available_providers():
            return 'ROCMExecutionProvider'
        else:
            return 'CUDAExecutionProvider'
    return 'CPUExecutionProvider'