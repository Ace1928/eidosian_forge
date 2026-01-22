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
def create_uri_cache_for_plugin(self, plugin: RuntimeEnvPlugin) -> URICache:
    """Create a URI cache for a plugin.

        Args:
            plugin_name: The name of the plugin.

        Returns:
            The created URI cache for the plugin.
        """
    cache_size_env_var = f'RAY_RUNTIME_ENV_{plugin.name}_CACHE_SIZE_GB'.upper()
    cache_size_bytes = int(1024 ** 3 * float(os.environ.get(cache_size_env_var, 10)))
    return URICache(plugin.delete_uri, cache_size_bytes)