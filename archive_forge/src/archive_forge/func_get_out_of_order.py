from abc import ABC
from dataclasses import dataclass
from queue import Empty as QueueEmpty
from queue import Queue
from typing import Dict, List, Optional
import torch
from fairscale.internal.object import pyobject_to_tensor, tensor_to_pyobject
from fairscale.nn.model_parallel import get_pipeline_parallel_group
from .types import MESSAGE_GENERATION_START, InputDevice, PipeMessage, Tensors
def get_out_of_order(self, queue_name: int, index: int) -> Tensors:
    """Receive a message with a known microbatch index, and handle out-of-order
        messages by placing them back on the queue"""
    message = self.recv_message(queue_name)
    assert message.args == index
    return message.tensors