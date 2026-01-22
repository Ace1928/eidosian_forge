from __future__ import annotations
import logging
import operator
import types
from typing import (
import torch
import torch._ops
import torch.fx
from torch.onnx._internal import _beartype
from torch.onnx._internal.fx import (
from onnxscript.function_libs.torch_lib import (  # type: ignore[import]
Separate Python args and kwargs into ONNX inputs and attributes.

        Extra_kwargs are ignored if their values are None. For example, if the
        OpSchema has an attribute "rounding_mode" and the caller provides
        "rounding_mode=None", the attribute "rounding_mode" will not be included
        in the returned attributes when the OnnxFunction signature doesn't have
        "rounding_mode" as an attribute.

        Args:
            param_schemas: The parameter schemas of an Op or a OnnxFunction.
            args: The Python positional arguments supplied by the caller.
            kwargs: The Python keyword arguments supplied by the caller.
            fill_defaults: Whether to fill the default values for attributes.

        Returns:
            A tuple of two elements:
            - A list of ONNX inputs.
            - An dictionary of ONNX attribute names and values.

        Raises:
            TypeError: When allow_extra_kwargs is False and there are unknown kwargs.
            TypeError: When a required input is not provided.
        