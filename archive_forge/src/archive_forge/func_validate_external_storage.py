import atexit
import collections
import datetime
import errno
import json
import logging
import os
import random
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from collections import defaultdict
from typing import Dict, Optional, Tuple, IO, AnyStr
from filelock import FileLock
import ray
import ray._private.ray_constants as ray_constants
import ray._private.services
from ray._private import storage
from ray._raylet import GcsClient, get_session_key_from_storage
from ray._private.resource_spec import ResourceSpec
from ray._private.services import serialize_config, get_address
from ray._private.utils import open_log, try_to_create_directory, try_to_symlink
def validate_external_storage(self):
    """Make sure we can setup the object spilling external storage.
        This will also fill up the default setting for object spilling
        if not specified.
        """
    object_spilling_config = self._config.get('object_spilling_config', {})
    automatic_spilling_enabled = self._config.get('automatic_object_spilling_enabled', True)
    if not automatic_spilling_enabled:
        return
    if not object_spilling_config:
        object_spilling_config = os.environ.get('RAY_object_spilling_config', '')
    if not object_spilling_config:
        object_spilling_config = json.dumps({'type': 'filesystem', 'params': {'directory_path': self._session_dir}})
    deserialized_config = json.loads(object_spilling_config)
    self._ray_params._system_config['object_spilling_config'] = object_spilling_config
    self._config['object_spilling_config'] = object_spilling_config
    is_external_storage_type_fs = deserialized_config['type'] == 'filesystem'
    self._ray_params._system_config['is_external_storage_type_fs'] = is_external_storage_type_fs
    self._config['is_external_storage_type_fs'] = is_external_storage_type_fs
    from ray._private import external_storage
    external_storage.setup_external_storage(deserialized_config, self._session_name)
    external_storage.reset_external_storage()