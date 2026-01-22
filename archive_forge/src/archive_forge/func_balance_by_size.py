from typing import Any, List, Union, Sequence
import torch
from torch import Tensor
import torch.nn as nn
from . import blockpartition
from .profile import profile_sizes, profile_times
def balance_by_size(partitions: int, module: nn.Sequential, input: Union[List[Any], Tensor], *, chunks: int=1, param_scale: float=2.0, device: Device=torch.device('cuda')) -> List[int]:
    """Naive automatic balancing by CUDA memory usage per layer.

    During training, required memory for parameters depends on which optimizer
    is used. Optimizers may use buffers for each parameter to track
    optimization statistics internally, such as momentum buffer in SGD.

    To get more reliable size based balance, you should specify `param_scale`
    with regard to your optimizer. The default `param_scale` is 2 instead of 1
    due to gradient accumulation which is necessary for every optimizer.

    Follow this guide to choose correct `param_scale` for typical optimizers:

    =========  =============  =========================================
    Optimizer  `param_scale`  Internal State
    =========  =============  =========================================
    SGD        2--3           (momentum_buffer)
    Adam       4--5           exp_avg, exp_avg_sq, (max_exp_avg_sq)
    Adadelta   4              square_avg, acc_delta
    Adagrad    3              sum
    RMSprop    3--5           square_avg, (momentum_buffer), (grad_avg)
    =========  =============  =========================================

    Here's a simple example with the Adam optimizer::

        balance = balance_by_size(
            torch.cuda.device_count(),
            model,

            # Same size with mini-batch to train
            torch.empty(1024, 3, 224, 224),

            # Number of micro-batches to train with Pipe
            chunks=8,

            # 4 for Adam
            param_scale=4.0,
        )

        pipe = Pipe(model, balance, chunks=8)
        adam = Adam(pipe.parameters())

    Args:
        partitions (int):
            intended number of partitions
        module (torch.nn.Sequential):
            sequential module to be partitioned
        input (torch.Tensor):
            example mini-batch with the same size to train

    Keyword Args:
        chunks (int):
            number of micro-batches will be used to train (default: ``1``)
        param_scale (float):
            how many copies of parameters would be allocated for training. It
            depends on optimizer. See the above guide. (default: ``2.0``)
        device ('cuda' device):
            CUDA device where each layer is profiled (default: the current CUDA
            device)

    Returns:
        A list of number of layers in each partition. Use it for the `balance`
        parameter of :class:`~torchpipe.Pipe`.

    .. note::
        `module` and `input` must be placed on the same CUDA device.

    """
    sizes = profile_sizes(module, input, chunks, param_scale, torch.device(device))
    return balance_cost(sizes, partitions)