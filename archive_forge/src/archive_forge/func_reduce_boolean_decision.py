from typing import Any, Dict, List, Optional, Union
import torch
from torch import Tensor
from torch.nn import DataParallel, Module
from typing_extensions import override
from lightning_fabric.accelerators import Accelerator
from lightning_fabric.plugins.io.checkpoint_io import CheckpointIO
from lightning_fabric.plugins.precision import Precision
from lightning_fabric.strategies.parallel import ParallelStrategy
from lightning_fabric.strategies.registry import _StrategyRegistry
from lightning_fabric.strategies.strategy import TBroadcast, TReduce
from lightning_fabric.utilities.apply_func import apply_to_collection
from lightning_fabric.utilities.distributed import ReduceOp
@override
def reduce_boolean_decision(self, decision: bool, all: bool=True) -> bool:
    return decision