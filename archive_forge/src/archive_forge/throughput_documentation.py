from collections import deque
from typing import TYPE_CHECKING, Any, Callable, Deque, Dict, List, Optional, TypeVar, Union
import torch
from typing_extensions import override
from lightning_fabric.utilities.imports import _TORCH_GREATER_EQUAL_2_1
from lightning_fabric.utilities.rank_zero import rank_zero_only, rank_zero_warn
See :meth:`Throughput.compute`

        Args:
            step: Can be used to override the logging step.
            \**kwargs: See available parameters in :meth:`Throughput.compute`

        