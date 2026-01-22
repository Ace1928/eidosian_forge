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
def merge_resources(env_dict, params_dict):
    """Separates special case params and merges two dictionaries, picking from the
            first in the event of a conflict. Also emit a warning on every
            conflict.
            """
    num_cpus = env_dict.pop('CPU', None)
    num_gpus = env_dict.pop('GPU', None)
    memory = env_dict.pop('memory', None)
    object_store_memory = env_dict.pop('object_store_memory', None)
    result = params_dict.copy()
    result.update(env_dict)
    for key in set(env_dict.keys()).intersection(set(params_dict.keys())):
        if params_dict[key] != env_dict[key]:
            logger.warning(f'Autoscaler is overriding your resource:{key}: {params_dict[key]} with {env_dict[key]}.')
    return (num_cpus, num_gpus, memory, object_store_memory, result)