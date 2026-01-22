import copy
from enum import Enum
from typing import Any, Dict, List
from ray.autoscaler._private.util import hash_runtime_conf, prepare_config
def get_node_resources(self, instance_type_name: str) -> Dict[str, float]:
    return copy.deepcopy(self._node_configs['available_node_types'][instance_type_name].get('resources', {}))