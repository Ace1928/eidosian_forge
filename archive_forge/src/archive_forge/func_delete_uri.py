import logging
import os
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, List, Optional
from ray._private.runtime_env.context import RuntimeEnvContext
from ray._private.runtime_env.packaging import (
from ray._private.runtime_env.plugin import RuntimeEnvPlugin
from ray._private.runtime_env.working_dir import set_pythonpath_in_context
from ray._private.utils import get_directory_size_bytes, try_to_create_directory
from ray.exceptions import RuntimeEnvSetupError
def delete_uri(self, uri: str, logger: Optional[logging.Logger]=default_logger) -> int:
    """Delete URI and return the number of bytes deleted."""
    logger.info('Got request to delete pymodule URI %s', uri)
    local_dir = get_local_dir_from_uri(uri, self._resources_dir)
    local_dir_size = get_directory_size_bytes(local_dir)
    deleted = delete_package(uri, self._resources_dir)
    if not deleted:
        logger.warning(f'Tried to delete nonexistent URI: {uri}.')
        return 0
    return local_dir_size