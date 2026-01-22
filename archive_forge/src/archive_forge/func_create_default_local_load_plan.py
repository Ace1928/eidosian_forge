import dataclasses
import io
import logging
import operator
from collections import ChainMap
from functools import reduce
from typing import List, Tuple, Dict, Any, Union, cast
import torch
from torch.distributed._shard._utils import narrow_tensor_by_index
from torch.distributed._tensor import DTensor
from torch.distributed.checkpoint.planner import (
from torch.distributed.checkpoint.metadata import (
from torch.distributed.checkpoint.planner_helpers import (
from torch.distributed.checkpoint._nested_dict import (
from torch.distributed.checkpoint._sharded_tensor_utils import (
from torch.distributed.checkpoint._dedup_tensors import dedup_tensors
from torch.distributed.checkpoint.utils import find_state_dict_object
from torch.distributed.checkpoint._traverse import set_element
def create_default_local_load_plan(state_dict: Dict[str, Any], metadata: Metadata) -> LoadPlan:
    requests = []
    '\n    Create the ``LoadPlan`` used by DefaultLoadPlanner.\n\n    It produces one read item per value in ``state_dict`` using the metadata in ``metadata``.\n\n    The default behavior is to match key exactly between state_dict and metadata.\n    It handles resharding by issuing multiple read requests against storage in order to match\n    load requirements.\n    '
    for fqn, obj in state_dict.items():
        md = metadata.state_dict_metadata[fqn]
        if isinstance(obj, DTensor):
            if obj.device_mesh.get_coordinate() is not None:
                requests += _create_read_items(fqn, md, obj)
        else:
            requests += _create_read_items(fqn, md, obj)
    return LoadPlan(requests)