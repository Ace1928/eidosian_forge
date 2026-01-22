import logging
import traceback
from typing import TYPE_CHECKING
import numpy as np
import torch
import onnxruntime as ort
from onnxruntime.capi.onnxruntime_inference_collection import OrtValue
from onnxruntime.transformers.io_binding_helper import TypeHelper as ORTTypeHelper
from ..utils import is_cupy_available, is_onnxruntime_training_available
@staticmethod
def to_pytorch_via_numpy(ort_value: OrtValue) -> torch.Tensor:
    ort_device = ort_value.device_name().lower()
    return torch.tensor(ort_value.numpy()).to(ort_device)