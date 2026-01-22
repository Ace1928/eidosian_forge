import copy
from enum import Enum
from typing import Any, Dict, List
from ray.autoscaler._private.util import hash_runtime_conf, prepare_config
@property
def runtime_hash(self) -> str:
    return self._runtime_hash