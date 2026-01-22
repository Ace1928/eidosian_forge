import base64
import collections
import errno
import io
import json
import logging
import mmap
import multiprocessing
import os
import random
import shutil
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional, IO, AnyStr
import psutil
from filelock import FileLock
import ray
import ray._private.ray_constants as ray_constants
from ray._raylet import GcsClient, GcsClientOptions
from ray.core.generated.common_pb2 import Language
from ray._private.ray_constants import RAY_NODE_IP_FILENAME
def get_cached_node_ip_address(session_dir: str) -> str:
    """Get a node address cached on this session.

    If a ray instance is started by `ray start --node-ip-address`,
    the node ip address is cached to a file RAY_NODE_IP_FILENAME.
    Otherwise, the file exists, but it is emptyl.

    This API is process-safe, meaning the file access is protected by
    a file lock.

    Args:
        session_dir: Path to the Ray session directory.

    Returns:
        node_ip_address cached on the current node. None if the node
        the file doesn't exist, meaning ray instance hasn't been
        started on a current node. If node_ip_address is not written
        to a file, it means --node-ip-address is not given, and in this
        case, we find the IP address ourselves.
    """
    file_path = Path(os.path.join(session_dir, RAY_NODE_IP_FILENAME))
    cached_node_ip_address = {}
    with FileLock(str(file_path.absolute()) + '.lock'):
        if not file_path.exists():
            return None
        with file_path.open() as f:
            cached_node_ip_address.update(json.load(f))
        if 'node_ip_address' in cached_node_ip_address:
            return cached_node_ip_address['node_ip_address']
        else:
            return ray.util.get_node_ip_address()