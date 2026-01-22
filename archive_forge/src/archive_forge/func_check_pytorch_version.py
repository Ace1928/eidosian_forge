from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union
import torch
from torch import Tensor, nn
from torch.distributed import rpc
from fairscale.internal import torch_version
from fairscale.nn.pipe import microbatch
from .data import DataConsumer
from .graph import Node, PipelineModulesGraph
from .partition_handler import DistributedPipelineRecord, PartitionHandler
def check_pytorch_version() -> None:
    if torch_version() < (1, 8, 0):
        raise Exception('DistributedPipeline requires PyTorch version 1.8 or higher')