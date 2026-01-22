import contextlib
import copy
import functools
import inspect
import os
import re
import warnings
from typing import Any, Callable, Dict, List, Optional, Set, TypeVar
from typing_extensions import ParamSpec
import torch
from torch._jit_internal import (
from torch.autograd import function
from torch.jit._script import _CachedForward, script, ScriptModule
from torch.jit._state import _enabled, _python_cu
from torch.nn import Module
from torch.testing._comparison import default_tolerances
def compare_outputs(original, reference, match_what):
    all_ok = True
    for i, (orig, ref) in enumerate(zip(original, reference)):
        try:
            if orig.is_quantized:
                orig = orig.dequantize()
            if ref.is_quantized:
                ref = ref.dequantize()
            if orig.is_mkldnn:
                orig = orig.to_dense()
            if ref.is_mkldnn:
                ref = ref.to_dense()
            if ref.is_complex() or orig.is_complex():
                torch.testing.assert_close(orig.to(torch.cdouble), ref.to(torch.cdouble), rtol=check_tolerance, atol=default_tolerances(orig, ref)[1], equal_nan=True)
            elif orig.is_mps or ref.is_mps:
                torch.testing.assert_close(orig.float(), ref.float(), rtol=check_tolerance, atol=default_tolerances(orig, ref)[1], equal_nan=True)
            else:
                torch.testing.assert_close(orig.double(), ref.double(), rtol=check_tolerance, atol=default_tolerances(orig, ref)[1], equal_nan=True)
        except AssertionError as e:
            maybe_warn_nondeterministic()
            warnings.warn('Output nr ' + str(i + 1) + '. of the traced function does not match the corresponding output of the ' + match_what + '. Detailed error:\n' + str(e), category=TracerWarning, stacklevel=4)
            all_ok = False
    return all_ok