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
def canonicalize_bootstrap_address_or_die(addr: str, temp_dir: Optional[str]=None) -> str:
    """Canonicalizes Ray cluster bootstrap address to host:port.

    This function should be used when the caller expects there to be an active
    and local Ray instance. If no address is provided or address="auto", this
    will autodetect the latest Ray instance created with `ray start`.

    For convenience, if no address can be autodetected, this function will also
    look for any running local GCS processes, based on pgrep output. This is to
    allow easier use of Ray CLIs when debugging a local Ray instance (whose GCS
    addresses are not recorded).

    Returns:
        Ray cluster address string in <host:port> format. Throws a
        ConnectionError if zero or multiple active Ray instances are
        autodetected.
    """
    bootstrap_addr = canonicalize_bootstrap_address(addr, temp_dir=temp_dir)
    if bootstrap_addr is not None:
        return bootstrap_addr
    running_gcs_addresses = find_gcs_addresses()
    if len(running_gcs_addresses) == 0:
        raise ConnectionError('Could not find any running Ray instance. Please specify the one to connect to by setting the `--address` flag or `RAY_ADDRESS` environment variable.')
    if len(running_gcs_addresses) > 1:
        raise ConnectionError(f'Found multiple active Ray instances: {running_gcs_addresses}. Please specify the one to connect to by setting the `--address` flag or `RAY_ADDRESS` environment variable.')
    return running_gcs_addresses.pop()