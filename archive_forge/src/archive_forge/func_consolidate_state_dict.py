import collections
import copy
import enum
import inspect
import io
import logging
from itertools import chain
from typing import Any, Callable, Dict, List, Optional, Set, Type, Union
import torch
import torch.distributed as dist
from torch.distributed.algorithms.join import Join, Joinable, JoinHook
from torch.distributed.optim.utils import functional_optim_map
from torch.optim import Optimizer
def consolidate_state_dict(self, to: int=0) -> None:
    """
        Consolidate a list of ``state_dict`` s (one per rank) on the target rank.

        Arguments:
            to (int): the rank that receives the optimizer states (default: 0).

        Raises:
            RuntimeError: if ``overlap_with_ddp=True`` and this method is
                called before this :class:`ZeroRedundancyOptimizer` instance
                has been fully initialized, which happens once
                :class:`DistributedDataParallel` gradient buckets have been
                rebuilt.

        .. warning:: This needs to be called on all ranks.
        """
    self._check_overlap_initialized()
    self._sync_param_groups(self.param_groups, self.optim.param_groups)
    empty_messenger = torch.tensor([0], dtype=torch.uint8, device=self._default_device)
    self._all_state_dicts = []
    for rank in range(self.world_size):
        global_rank = dist.distributed_c10d.get_global_rank(self.process_group, rank)
        if self.rank == to:
            if rank == self.rank:
                self._all_state_dicts.append(_recursive_copy_to_device(self.optim.state_dict(), non_blocking=True, device=torch.device('cpu')))
            else:
                local_state_dict = _broadcast_object(empty_messenger, src_rank=global_rank, group=self.process_group, device=self._default_device)
                self._all_state_dicts.append(_recursive_copy_to_device(local_state_dict, non_blocking=True, device=torch.device('cpu')))
        elif rank == self.rank:
            _ = _broadcast_object(self.optim.state_dict(), src_rank=self.global_rank, group=self.process_group, device=self._default_device)
        elif rank != to:
            _ = _broadcast_object(empty_messenger, src_rank=global_rank, group=self.process_group, device=self._default_device)