import warnings
from inspect import signature
from itertools import chain
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, List, Tuple, Union
import numpy as np
from packaging.version import Version, parse
from ..tokenization_utils_base import PreTrainedTokenizerBase
from ..utils import (
from .config import OnnxConfig
def check_onnxruntime_requirements(minimum_version: Version):
    """
    Check onnxruntime is installed and if the installed version match is recent enough

    Raises:
        ImportError: If onnxruntime is not installed or too old version is found
    """
    try:
        import onnxruntime
        ort_version = parse(onnxruntime.__version__)
        if ort_version < ORT_QUANTIZE_MINIMUM_VERSION:
            raise ImportError(f'We found an older version of onnxruntime ({onnxruntime.__version__}) but we require onnxruntime to be >= {minimum_version} to enable all the conversions options.\nPlease update onnxruntime by running `pip install --upgrade onnxruntime`')
    except ImportError:
        raise ImportError("onnxruntime doesn't seem to be currently installed. Please install the onnxruntime by running `pip install onnxruntime` and relaunch the conversion.")