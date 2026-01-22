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
def get_node_ip_address(address='8.8.8.8:53'):
    if ray._private.worker._global_node is not None:
        return ray._private.worker._global_node.node_ip_address
    if not ray_constants.ENABLE_RAY_CLUSTER:
        return '127.0.0.1'
    return node_ip_address_from_perspective(address)