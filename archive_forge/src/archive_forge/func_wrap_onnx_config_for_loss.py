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
def wrap_onnx_config_for_loss(onnx_config: OnnxConfig) -> OnnxConfig:
    return OnnxConfigWithLoss(onnx_config)