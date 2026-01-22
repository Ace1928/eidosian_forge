import os
import threading
from typing import TYPE_CHECKING, Any, Dict, List, Optional
import ray
from ray._private.ray_constants import env_integer
from ray.util.annotations import DeveloperAPI
from ray.util.scheduling_strategies import SchedulingStrategyT
def remove_config(self, key: str) -> None:
    """Remove a key-value style config.

        Args:
            key: The key of the config.
        """
    self._kv_configs.pop(key, None)