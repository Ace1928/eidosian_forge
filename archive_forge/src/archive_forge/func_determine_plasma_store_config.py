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
def determine_plasma_store_config(object_store_memory: int, plasma_directory: Optional[str]=None, huge_pages: bool=False):
    """Figure out how to configure the plasma object store.

    This will determine which directory to use for the plasma store. On Linux,
    we will try to use /dev/shm unless the shared memory file system is too
    small, in which case we will fall back to /tmp. If any of the object store
    memory or plasma directory parameters are specified by the user, then those
    values will be preserved.

    Args:
        object_store_memory: The object store memory to use.
        plasma_directory: The user-specified plasma directory parameter.
        huge_pages: The user-specified huge pages parameter.

    Returns:
        The plasma directory to use. If it is specified by the user, then that
            value will be preserved.
    """
    if not isinstance(object_store_memory, int):
        object_store_memory = int(object_store_memory)
    if huge_pages and (not (sys.platform == 'linux' or sys.platform == 'linux2')):
        raise ValueError('The huge_pages argument is only supported on Linux.')
    system_memory = ray._private.utils.get_system_memory()
    if plasma_directory is None:
        if sys.platform == 'linux' or sys.platform == 'linux2':
            shm_avail = ray._private.utils.get_shared_memory_bytes()
            if shm_avail > object_store_memory:
                plasma_directory = '/dev/shm'
            elif not os.environ.get('RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE') and object_store_memory > ray_constants.REQUIRE_SHM_SIZE_THRESHOLD:
                raise ValueError('The configured object store size ({} GB) exceeds /dev/shm size ({} GB). This will harm performance. Consider deleting files in /dev/shm or increasing its size with --shm-size in Docker. To ignore this warning, set RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE=1.'.format(object_store_memory / 1000000000.0, shm_avail / 1000000000.0))
            else:
                plasma_directory = ray._private.utils.get_user_temp_dir()
                logger.warning("WARNING: The object store is using {} instead of /dev/shm because /dev/shm has only {} bytes available. This will harm performance! You may be able to free up space by deleting files in /dev/shm. If you are inside a Docker container, you can increase /dev/shm size by passing '--shm-size={:.2f}gb' to 'docker run' (or add it to the run_options list in a Ray cluster config). Make sure to set this to more than 30% of available RAM.".format(ray._private.utils.get_user_temp_dir(), shm_avail, object_store_memory * 1.1 / 2 ** 30))
        else:
            plasma_directory = ray._private.utils.get_user_temp_dir()
        if object_store_memory > system_memory:
            raise ValueError('The requested object store memory size is greater than the total available memory.')
    else:
        plasma_directory = os.path.abspath(plasma_directory)
        logger.info('object_store_memory is not verified when plasma_directory is set.')
    if not os.path.isdir(plasma_directory):
        raise ValueError(f'The file {plasma_directory} does not exist or is not a directory.')
    if huge_pages and plasma_directory is None:
        raise ValueError('If huge_pages is True, then the plasma_directory argument must be provided.')
    if object_store_memory < ray_constants.OBJECT_STORE_MINIMUM_MEMORY_BYTES:
        raise ValueError('Attempting to cap object store memory usage at {} bytes, but the minimum allowed is {} bytes.'.format(object_store_memory, ray_constants.OBJECT_STORE_MINIMUM_MEMORY_BYTES))
    if sys.platform == 'darwin' and object_store_memory > ray_constants.MAC_DEGRADED_PERF_MMAP_SIZE_LIMIT and (os.environ.get('RAY_ENABLE_MAC_LARGE_OBJECT_STORE') != '1'):
        raise ValueError("The configured object store size ({:.4}GiB) exceeds the optimal size on Mac ({:.4}GiB). This will harm performance! There is a known issue where Ray's performance degrades with object store size greater than {:.4}GB on a Mac.To reduce the object store capacity, specify`object_store_memory` when calling ray.init() or ray start.To ignore this warning, set RAY_ENABLE_MAC_LARGE_OBJECT_STORE=1.".format(object_store_memory / 2 ** 30, ray_constants.MAC_DEGRADED_PERF_MMAP_SIZE_LIMIT / 2 ** 30, ray_constants.MAC_DEGRADED_PERF_MMAP_SIZE_LIMIT / 2 ** 30))
    logger.debug('Determine to start the Plasma object store with {} GB memory using {}.'.format(round(object_store_memory / 10 ** 9, 2), plasma_directory))
    return (plasma_directory, object_store_memory)