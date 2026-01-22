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
def patch_ops(self):
    for spec in self._patching_specs:
        custom_op = spec.custom_op if spec.op_wrapper is None else spec.op_wrapper(spec.custom_op)
        setattr(spec.o, spec.name, custom_op)