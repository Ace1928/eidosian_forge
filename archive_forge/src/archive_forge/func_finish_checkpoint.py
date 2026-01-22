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
def finish_checkpoint(all_results):
    assert global_metatadata is not None
    storage_writer.finish(metadata=global_metatadata, results=all_results)
    return global_metatadata