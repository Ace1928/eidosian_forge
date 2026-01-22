import logging
import os
import json
from abc import ABC
from typing import List, Dict, Optional, Any, Type
from ray._private.runtime_env.context import RuntimeEnvContext
from ray._private.runtime_env.uri_cache import URICache
from ray._private.runtime_env.constants import (
from ray.util.annotations import DeveloperAPI
from ray._private.utils import import_attr
def validate_priority(self, priority: Any) -> None:
    if not isinstance(priority, int) or priority < RAY_RUNTIME_ENV_PLUGIN_MIN_PRIORITY or priority > RAY_RUNTIME_ENV_PLUGIN_MAX_PRIORITY:
        raise RuntimeError(f'Invalid runtime env priority {priority}, it should be an integer between {RAY_RUNTIME_ENV_PLUGIN_MIN_PRIORITY} and {RAY_RUNTIME_ENV_PLUGIN_MAX_PRIORITY}.')