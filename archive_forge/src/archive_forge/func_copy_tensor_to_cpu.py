from typing import Any, Dict, List, Union
from utils.log import quick_log
from fastapi import APIRouter, HTTPException, Request, Response, status
from pydantic import BaseModel
import gc
import copy
import global_var
def copy_tensor_to_cpu(tensors):
    import torch
    import numpy as np
    devices: List[torch.device] = []
    copied: Union[Any, None] = None
    tensors_type = type(tensors)
    if tensors_type == list:
        if hasattr(tensors[0], 'device'):
            devices = [tensor.device for tensor in tensors]
            copied = [tensor.cpu() for tensor in tensors]
        else:
            copied = tensors
    elif tensors_type == torch.Tensor:
        devices = [tensors.device]
        copied = tensors.cpu()
    elif tensors_type == np.ndarray:
        copied = tensors
    else:
        copied = tensors.back()
    return (copied, devices)