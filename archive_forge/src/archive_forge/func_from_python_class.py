from __future__ import annotations
import abc
import contextlib
import dataclasses
import difflib
import io
import logging
import sys
from typing import Any, Callable, Optional, Tuple
import torch
import torch.fx
from torch._subclasses import fake_tensor
from torch.fx.experimental.proxy_tensor import maybe_disable_fake_tensor_mode
from torch.onnx._internal import _beartype
from torch.onnx._internal.fx import diagnostics, onnxfunction_dispatcher
@classmethod
def from_python_class(cls, python_class: type) -> PackageInfo:
    package_name = python_class.__module__.split('.')[0]
    package = __import__(package_name)
    version = getattr(package, '__version__', None)
    commit_hash = None
    return cls(package_name, version, commit_hash)