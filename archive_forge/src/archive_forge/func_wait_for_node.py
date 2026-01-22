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
def wait_for_node(gcs_address: str, node_plasma_store_socket_name: str, timeout: int=_timeout):
    """Wait until this node has appeared in the client table.

    Args:
        gcs_address: The gcs address
        node_plasma_store_socket_name: The
            plasma_store_socket_name for the given node which we wait for.
        timeout: The amount of time in seconds to wait before raising an
            exception.

    Raises:
        TimeoutError: An exception is raised if the timeout expires before
            the node appears in the client table.
    """
    gcs_options = GcsClientOptions.from_gcs_address(gcs_address)
    global_state = ray._private.state.GlobalState()
    global_state._initialize_global_state(gcs_options)
    start_time = time.time()
    while time.time() - start_time < timeout:
        clients = global_state.node_table()
        object_store_socket_names = [client['ObjectStoreSocketName'] for client in clients]
        if node_plasma_store_socket_name in object_store_socket_names:
            return
        else:
            time.sleep(0.1)
    raise TimeoutError(f'Timed out after {timeout} seconds while waiting for node to startup. Did not find socket name {node_plasma_store_socket_name} in the list of object store socket names.')