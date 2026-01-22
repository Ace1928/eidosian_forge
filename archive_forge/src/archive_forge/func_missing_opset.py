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
def missing_opset(package_name: str):
    message = f'The installed `{package_name}` does not support the specified ONNX opset version {opset_version}. Install a newer `{package_name}` package or specify an older opset version.'
    log.fatal(message)
    return UnsatisfiedDependencyError(package_name, message)