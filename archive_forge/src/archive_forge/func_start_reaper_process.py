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
def start_reaper_process(self):
    """
        Start the reaper process.

        This must be the first process spawned and should only be called when
        ray processes should be cleaned up if this process dies.
        """
    assert not self.kernel_fate_share, 'a reaper should not be used with kernel fate-sharing'
    process_info = ray._private.services.start_reaper(fate_share=False)
    assert ray_constants.PROCESS_TYPE_REAPER not in self.all_processes
    if process_info is not None:
        self.all_processes[ray_constants.PROCESS_TYPE_REAPER] = [process_info]