from threading import Event, Lock, Thread
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, cast
import torch
from torch import nn
from torch.distributed import ProcessGroup, rpc
from torch.distributed.distributed_c10d import _get_global_rank
from fairscale.nn.model_parallel.initialize import get_pipeline_parallel_group
from .async_pipe import AsyncPipe
from .types import EVENT_LOOP_QUEUE, PipeMessage, TensorOrTensors
def callback_with_model(callback: Callable[[Any, AsyncPipe], None], ctx: Any) -> None:
    try:
        group = get_pipeline_parallel_group()
        set_device_based_on_group(group)
        with PipeModel.lock:
            callback(ctx, PipeModel)
    except Exception as e:
        print(f'callback_with_model got {e}')