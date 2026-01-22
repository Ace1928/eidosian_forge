import logging
import os
from typing import List
import numpy as np
import ray
from ray.util.collective import types
def destroy_collective_group(self, group_name):
    """Group destructor."""
    if not self.is_group_exist(group_name):
        logger.warning("The group '{}' does not exist.".format(group_name))
        return
    g = self._name_group_map[group_name]
    del self._group_name_map[g]
    del self._name_group_map[group_name]
    g.destroy_group()
    name = 'info_' + group_name
    try:
        store = ray.get_actor(name)
        ray.kill(store)
    except ValueError:
        pass