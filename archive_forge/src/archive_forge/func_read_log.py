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
def read_log(filename, lines_to_read):
    """Read a log file and return the last 20 lines."""
    dashboard_log = os.path.join(logdir, filename)
    lines_to_read = 20
    lines = []
    with open(dashboard_log, 'rb') as f:
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            end = mm.size()
            for _ in range(lines_to_read):
                sep = mm.rfind(b'\n', 0, end - 1)
                if sep == -1:
                    break
                lines.append(mm[sep + 1:end].decode('utf-8'))
                end = sep
    lines.append(f'The last {lines_to_read} lines of {dashboard_log} (it contains the error message from the dashboard): ')
    return lines