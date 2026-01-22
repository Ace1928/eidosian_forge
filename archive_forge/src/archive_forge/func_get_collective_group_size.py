import logging
import os
from typing import List
import numpy as np
import ray
from ray.util.collective import types
def get_collective_group_size(group_name: str='default') -> int:
    """Return the size of the collective group with the given name.

    Args:
        group_name: the name of the group to query

    Returns:
        The world size of the collective group, -1 if the group does
            not exist or the process does not belong to the group.
    """
    _check_inside_actor()
    if not is_group_initialized(group_name):
        return -1
    g = _group_mgr.get_group_by_name(group_name)
    return g.world_size