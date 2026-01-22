import copy
from enum import Enum
from typing import Any, Dict, List
from ray.autoscaler._private.util import hash_runtime_conf, prepare_config
@property
def restart_only(self) -> bool:
    return self._node_configs.get('restart_only', False)