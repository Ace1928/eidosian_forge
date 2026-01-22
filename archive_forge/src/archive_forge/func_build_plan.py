import abc
import torch.nn as nn
from dataclasses import dataclass
from typing import Dict, List, Optional, Union
from torch.distributed._shard.sharder import Sharder
from torch.distributed._shard.sharding_spec import ShardingSpec
@abc.abstractmethod
def build_plan(self, module: nn.Module) -> ShardingPlan:
    """
        Given a nn.Module, define how to shard the module across
        ranks, return a ShardingPlan
        Args:
            module (:class:`torch.nn.Module`):
                The module to apply sharding to.
        Returns:
            A :class:`torch.distributed._shard.sharding_plan.ShardingPlan` object that
            represents how to shard the module.
        """
    pass