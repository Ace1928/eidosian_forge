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
def write_node_ip_address(session_dir: str, node_ip_address: Optional[str]) -> None:
    """Write a node ip address of the current session to
    RAY_NODE_IP_FILENAME.

    If a ray instance is started by `ray start --node-ip-address`,
    the node ip address is cached to a file RAY_NODE_IP_FILENAME.

    This API is process-safe, meaning the file access is protected by
    a file lock.

    The file contains a single string node_ip_address. If nothing
    is written, it means --node-ip-address was not given, and Ray
    resolves the IP address on its own. It assumes in a single node,
    you can have only 1 IP address (which is the assumption ray
    has in general).

    node_ip_address is the ip address of the current node.

    Args:
        session_dir: The path to Ray session directory.
        node_ip_address: The node IP address of the current node.
            If None, it means the node ip address is not given
            by --node-ip-address. In this case, we don't write
            anything to a file.
    """
    file_path = Path(os.path.join(session_dir, RAY_NODE_IP_FILENAME))
    cached_node_ip_address = {}
    with FileLock(str(file_path.absolute()) + '.lock'):
        if not file_path.exists():
            with file_path.open(mode='w') as f:
                json.dump({}, f)
        with file_path.open() as f:
            cached_node_ip_address.update(json.load(f))
        cached_node_ip = cached_node_ip_address.get('node_ip_address')
        if node_ip_address is not None:
            if cached_node_ip:
                if cached_node_ip == node_ip_address:
                    return
                else:
                    logger.warning(f"The node IP address of the current host recorded in {RAY_NODE_IP_FILENAME} ({cached_node_ip}) is different from the current IP address: {node_ip_address}. Ray will use {node_ip_address} as the current node's IP address. Creating 2 instances in the same host with different IP address is not supported. Please create an enhnacement request tohttps://github.com/ray-project/ray/issues.")
            cached_node_ip_address['node_ip_address'] = node_ip_address
            with file_path.open(mode='w') as f:
                json.dump(cached_node_ip_address, f)