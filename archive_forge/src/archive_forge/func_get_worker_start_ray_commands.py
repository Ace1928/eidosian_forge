import copy
from enum import Enum
from typing import Any, Dict, List
from ray.autoscaler._private.util import hash_runtime_conf, prepare_config
def get_worker_start_ray_commands(self, num_successful_updates: int=0) -> List[str]:
    if num_successful_updates > 0 and (not self._node_config_provider.restart_only):
        return []
    return self._node_configs.get('worker_start_ray_commands', [])