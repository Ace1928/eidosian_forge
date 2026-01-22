import os
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple, Union, overload
import torch
import torch.distributed as dist
import torch.multiprocessing.reductions
from .. import _is_triton_available
from .common import BaseOperator, get_xformers_operator, register_operator
from .ipc import init_ipc
def make_stream_factory(self, current_stream: torch.cuda.Stream) -> Callable[[], torch.cuda.Stream]:

    def result():
        stream = [current_stream, self.second_stream][self.next_stream_idx]
        self.next_stream_idx += 1
        self.next_stream_idx %= 2
        return stream
    return result