from abc import ABC
from dataclasses import dataclass
from queue import Empty as QueueEmpty
from queue import Queue
from typing import Dict, List, Optional
import torch
from fairscale.internal.object import pyobject_to_tensor, tensor_to_pyobject
from fairscale.nn.model_parallel import get_pipeline_parallel_group
from .types import MESSAGE_GENERATION_START, InputDevice, PipeMessage, Tensors
def to_input_device(tensors: Tensors, input_device: InputDevice) -> Tensors:
    if input_device is None:
        return tensors
    else:
        return tuple((t.to(input_device) for t in tensors))