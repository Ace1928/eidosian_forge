import copy
from enum import Enum
from typing import Any, Dict, List
from ray.autoscaler._private.util import hash_runtime_conf, prepare_config
def get_head_setup_commands(self) -> List[str]:
    return self._node_configs.get('head_setup_commands', [])