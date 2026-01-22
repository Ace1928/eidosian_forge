from __future__ import (  # for onnx.ModelProto (ONNXProgram) and onnxruntime (ONNXRuntimeOptions)
import abc
import contextlib
import dataclasses
import io
import logging
import os
import warnings
from collections import defaultdict
from typing import (
from typing_extensions import Self
import torch
import torch._ops
import torch.export as torch_export
import torch.utils._pytree as pytree
from torch._subclasses import fake_tensor
from torch.onnx._internal import _beartype, io_adapter
from torch.onnx._internal.diagnostics import infra
from torch.onnx._internal.fx import (
def missing_package(package_name: str, exc_info: logging._ExcInfoType):
    message = f'Please install the `{package_name}` package (e.g. `python -m pip install {package_name}`).'
    log.fatal(message, exc_info=exc_info)
    return UnsatisfiedDependencyError(package_name, message)