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
def to_pytorch_via_cupy(ort_value: OrtValue) -> torch.Tensor:
    ort_device = ort_value.device_name().lower()
    if ort_device != 'cuda':
        raise RuntimeError(f'Exchange tensors to PyTorch via CuPy only when device is CUDA, got: {ort_device}')
    ort_type = ort_value.data_type()
    numpy_type = TypeHelper.ort_type_to_numpy_type(ort_type)
    memory = cp.cuda.UnownedMemory(ort_value.data_ptr(), 0, None)
    memory_ptr = cp.cuda.MemoryPointer(memory, 0)
    cp_array = cp.ndarray(shape=ort_value.shape(), memptr=memory_ptr, dtype=numpy_type)
    torch_tensor = torch.from_dlpack(cp_array.toDlpack())
    if 'bool' in ort_type:
        torch_tensor = torch_tensor.to(torch.bool)
    torch_tensor = torch_tensor.clone()
    return torch_tensor