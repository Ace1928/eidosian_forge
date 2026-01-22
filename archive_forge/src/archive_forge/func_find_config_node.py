import base64
import json
import os
from copy import deepcopy
from ..optimizer import AcceleratedOptimizer
from ..scheduler import AcceleratedScheduler
def find_config_node(self, ds_key_long):
    config = self.config
    nodes = ds_key_long.split('.')
    ds_key = nodes.pop()
    for node in nodes:
        config = config.get(node)
        if config is None:
            return (None, ds_key)
    return (config, ds_key)