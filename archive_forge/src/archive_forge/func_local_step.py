from typing import Optional
import warnings
import torch
import torch.distributed as dist
from torch.distributed.checkpoint.stateful import Stateful
from .planner import SavePlanner
from .default_planner import DefaultSavePlanner
from .storage import (
from .metadata import Metadata, STATE_DICT_TYPE
from .utils import _DistWrapper
def local_step():
    assert planner is not None
    planner.set_up_planner(state_dict, distW.is_coordinator)
    storage_writer.set_up_storage_writer(distW.is_coordinator)
    local_plan = planner.create_local_plan()
    local_plan = storage_writer.prepare_local_plan(local_plan)
    return local_plan