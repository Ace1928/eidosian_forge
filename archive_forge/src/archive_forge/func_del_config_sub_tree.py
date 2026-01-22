import base64
import json
import os
from copy import deepcopy
from ..optimizer import AcceleratedOptimizer
from ..scheduler import AcceleratedScheduler
def del_config_sub_tree(self, ds_key_long, must_exist=False):
    """
        Deletes a sub-section of the config file if it's found.

        Unless `must_exist` is `True` the section doesn't have to exist.
        """
    config = self.config
    nodes = ds_key_long.split('.')
    for node in nodes:
        parent_config = config
        config = config.get(node)
        if config is None:
            if must_exist:
                raise ValueError(f"Can't find {ds_key_long} entry in the config: {self.config}")
            else:
                return
    if parent_config is not None:
        parent_config.pop(node)