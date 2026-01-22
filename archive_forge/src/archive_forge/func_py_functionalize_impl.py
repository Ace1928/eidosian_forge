import contextlib
import ctypes
import importlib
import inspect
import sys
import types
from typing import Any, Callable, Dict, List, Type, Union
import torch._C
import torch.utils._pytree as pytree
from torch import _utils_internal
from torch._functorch.pyfunctorch import dispatch_functorch
def py_functionalize_impl(self, fn):
    from torch._subclasses.functional_tensor import CppFunctionalizeAPI as _CppFunctionalizeAPI, FunctorchFunctionalizeAPI as _FunctorchFunctionalizeAPI, PythonFunctionalizeAPI as _PythonFunctionalizeAPI

    def functionalize_dk_fn(*args, **kwargs):
        return fn(_CppFunctionalizeAPI(), *args, **kwargs)

    def functionalize_dispatch_mode_fn(mode, *args, **kwargs):
        return fn(_PythonFunctionalizeAPI(), *args, **kwargs)

    def functionalize_functorch_fn(interpreter, *args, **kwargs):
        return fn(_FunctorchFunctionalizeAPI(interpreter), *args, **kwargs)
    self.py_impl(torch._C.DispatchKey.Functionalize)(functionalize_dk_fn)
    self.py_impl(torch._subclasses.functional_tensor.FunctionalTensorMode)(functionalize_dispatch_mode_fn)
    self.py_impl(torch._C._functorch.TransformType.Functionalize)(functionalize_functorch_fn)
    return fn