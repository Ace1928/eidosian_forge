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
def parameter_rrefs(self) -> List[rpc.RRef]:
    remote_params = []
    for p in self.partitions:
        remote_params.extend(p.handler.rpc_sync().local_parameter_rrefs())
    return remote_params