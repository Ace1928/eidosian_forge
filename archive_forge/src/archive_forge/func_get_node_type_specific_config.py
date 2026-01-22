import copy
from enum import Enum
from typing import Any, Dict, List
from ray.autoscaler._private.util import hash_runtime_conf, prepare_config
def get_node_type_specific_config(self, instance_type_name: str, config_name: str) -> Any:
    config = self.get_config(config_name)
    node_specific_config = self._node_configs['available_node_types'].get(instance_type_name, {})
    if config_name in node_specific_config:
        config = node_specific_config[config_name]
    return config