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
def start_gcs_server(self):
    """Start the gcs server."""
    gcs_server_port = self._ray_params.gcs_server_port
    assert gcs_server_port > 0
    assert self._gcs_address is None, 'GCS server is already running.'
    assert self._gcs_client is None, 'GCS client is already connected.'
    stdout_file, stderr_file = self.get_log_file_handles('gcs_server', unique=True)
    process_info = ray._private.services.start_gcs_server(self.redis_address, self._logs_dir, self.session_name, stdout_file=stdout_file, stderr_file=stderr_file, redis_password=self._ray_params.redis_password, config=self._config, fate_share=self.kernel_fate_share, gcs_server_port=gcs_server_port, metrics_agent_port=self._ray_params.metrics_agent_port, node_ip_address=self._node_ip_address)
    assert ray_constants.PROCESS_TYPE_GCS_SERVER not in self.all_processes
    self.all_processes[ray_constants.PROCESS_TYPE_GCS_SERVER] = [process_info]
    self._gcs_address = f'{self._node_ip_address}:{gcs_server_port}'