from __future__ import annotations
import functools
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple, Union
import torch
import torch.fx
import torch.onnx
import torch.onnx._internal.fx.passes as passes
from torch.onnx._internal import _beartype, exporter, io_adapter
Generates a FX GraphModule using torch.fx.symbolic_trace API
    Args:
        concrete_args: Inputs to be partially specialized
            It can be used to remove control flow or data structures.
            For example::
                def f(a, b):
                    if b == True:
                        return a
                    else:
                        return a*2
            FX can typically not trace through this due to the presence of control
            flow. However, we can use `concrete_args` to specialize on the value of
            `b` to trace through this::
                f = fx.symbolic_trace(f, concrete_args={'b': False})
                assert f(3, False)  == 6
            Note that although you can still pass in different values of `b`, they will be ignored.
            It can also be used to eliminate data-structure handling from
            our function. This will use pytrees to flatten your input. To avoid
            overspecializing, pass in `fx.PH` for values that shouldn't be
            specialized. For example::
                def f(x):
                    out = 0
                    for v in x.values():
                        out += v
                    return out
                f = fx.symbolic_trace(f, concrete_args={'x': {'a': fx.PH, 'b': fx.PH, 'c': fx.PH}})
                assert f({'a': 1, 'b': 2, 'c': 4}) == 7
    