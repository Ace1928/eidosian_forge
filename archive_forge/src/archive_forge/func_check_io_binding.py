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
def check_io_binding(providers: List[str], use_io_binding: Optional[bool]=None) -> bool:
    """
    Whether to use IOBinding or not.
    """
    if use_io_binding is None and providers[0] == 'CUDAExecutionProvider':
        use_io_binding = True
    elif providers[0] != 'CPUExecutionProvider' and providers[0] != 'CUDAExecutionProvider':
        if use_io_binding is True:
            logger.warning('No need to enable IO Binding if the provider used is neither CPUExecutionProvider nor CUDAExecutionProvider. IO Binding will be turned off.')
        use_io_binding = False
    return use_io_binding