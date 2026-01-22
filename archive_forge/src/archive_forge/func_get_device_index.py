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
def get_device_index(device):
    if isinstance(device, str):
        device = torch.device(device)
    elif isinstance(device, int):
        return device
    return 0 if device.index is None else device.index