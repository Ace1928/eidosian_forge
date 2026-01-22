from abc import ABC
from dataclasses import dataclass
from queue import Empty as QueueEmpty
from queue import Queue
from typing import Dict, List, Optional
import torch
from fairscale.internal.object import pyobject_to_tensor, tensor_to_pyobject
from fairscale.nn.model_parallel import get_pipeline_parallel_group
from .types import MESSAGE_GENERATION_START, InputDevice, PipeMessage, Tensors
def recv_message_header(self, queue_name: int, nowait: bool=False) -> PipeMessage:
    if nowait:
        raise QueueEmpty
    tensor = torch.empty(MESSAGE_TENSOR_SIZE, dtype=torch.uint8, device=self.input_device)
    torch.cuda.current_stream().synchronize()
    torch.distributed.recv(tensor, src=None, tag=queue_name, group=get_pipeline_parallel_group())
    torch.cuda.current_stream().synchronize()
    return tensor_to_pyobject(tensor)