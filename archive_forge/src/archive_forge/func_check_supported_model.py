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
@classmethod
def check_supported_model(cls, model_type: str):
    if model_type not in cls._conf:
        model_types = ', '.join(cls._conf.keys())
        raise KeyError(f'{model_type} model type is not supported yet. Only {model_types} are supported. If you want to support {model_type} please propose a PR or open up an issue.')