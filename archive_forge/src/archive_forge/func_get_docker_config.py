import copy
from enum import Enum
from typing import Any, Dict, List
from ray.autoscaler._private.util import hash_runtime_conf, prepare_config
def get_docker_config(self, instance_type_name: str) -> Dict[str, Any]:
    docker_config = copy.deepcopy(self._node_configs.get('docker', {}))
    node_specific_docker_config = self._node_configs['available_node_types'][instance_type_name].get('docker', {})
    docker_config.update(node_specific_docker_config)
    return docker_config