from __future__ import annotations
import dataclasses
import re
import typing
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple, Union
import torch
from torch import _C
from torch._C import _onnx as _C_onnx
from torch.onnx._globals import GLOBALS
from torch.onnx._internal import _beartype, registration
@_beartype.beartype
def onnxscript_op(self, onnx_fn, *raw_args: Union[torch.Tensor, _C.Value], outputs: int=1, **kwargs):
    """Creates an ONNX operator from onnx-script function, taking "raw_args" as inputs and "kwargs" as attributes.

        onnx-script repository: https://github.com/microsoft/onnx-script

        Args:
            onnx_fn: ONNXFunction from onnx-script; An example can be found at
                https://github.com/microsoft/onnx-script#example
            raw_args: The inputs to the operator; usually provided
                as arguments to the `symbolic` definition.
            outputs: The number of outputs this operator returns.
                By default an operator is assumed to return a single output.
                If `outputs` is greater than one, this functions returns a tuple
                of output `Value`, representing each output of the ONNX operator
                in order.
            kwargs: The attributes of the ONNX operator, whose keys are named
                according to the following convention: `alpha_f` indicates
                the `alpha` attribute with type `f`.  The valid type specifiers are
                `f` (float), `i` (int), `s` (string) or `t` (Tensor).  An attribute
                specified with type float accepts either a single float, or a
                list of floats (e.g., you would say `dims_i` for a `dims` attribute
                that takes a list of integers).

        Returns:
            The value representing the single output of this operator (see the `outputs`
            keyword argument for multi-return nodes).
        """
    symbolic_name = f'{onnx_fn.opset.domain}::{onnx_fn.name}'
    opset_version = onnx_fn.opset.version
    registration.custom_onnx_symbolic(symbolic_name, opset_version)(onnx_fn)
    return _add_op(self, symbolic_name, *raw_args, outputs=outputs, **kwargs)