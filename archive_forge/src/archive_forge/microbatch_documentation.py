import typing
from typing import Any, Callable, List, Union, cast, Sequence
import torch
from torch import Tensor
import torch.cuda.comm
Calls a function on the microbatch. It also wraps
        the output with :class:`Batch`.
        