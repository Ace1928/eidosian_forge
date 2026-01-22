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
def ort_type_to_torch_type(ort_type: str):
    ort_type_to_torch_type_map = {'tensor(int64)': torch.int64, 'tensor(int32)': torch.int32, 'tensor(int8)': torch.int8, 'tensor(float)': torch.float32, 'tensor(float16)': torch.float16, 'tensor(bool)': torch.bool}
    if ort_type in ort_type_to_torch_type_map:
        return ort_type_to_torch_type_map[ort_type]
    else:
        raise ValueError(f'{ort_type} is not supported. Here is a list of supported data type: {ort_type_to_torch_type_map.keys()}')