import copy
import dataclasses
import warnings
from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterable, List, Mapping, Optional, Tuple, Union
import numpy as np
from packaging import version
from ..utils import TensorType, is_torch_available, is_vision_available, logging
from .utils import ParameterFormat, compute_effective_axis_dimension, compute_serialized_parameters_size
@property
def is_torch_support_available(self) -> bool:
    """
        The minimum PyTorch version required to export the model.

        Returns:
            `bool`: Whether the installed version of PyTorch is compatible with the model.
        """
    if is_torch_available():
        from transformers.utils import get_torch_version
        return version.parse(get_torch_version()) >= self.torch_onnx_minimum_version
    else:
        return False