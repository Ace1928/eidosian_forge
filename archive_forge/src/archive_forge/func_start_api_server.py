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
def start_api_server(self, *, include_dashboard: Optional[bool], raise_on_failure: bool):
    """Start the dashboard.

        Args:
            include_dashboard: If true, this will load all dashboard-related modules
                when starting the API server. Otherwise, it will only
                start the modules that are not relevant to the dashboard.
            raise_on_failure: If true, this will raise an exception
                if we fail to start the API server. Otherwise it will print
                a warning if we fail to start the API server.
        """
    _, stderr_file = self.get_log_file_handles('dashboard', unique=True, create_out=False)
    self._webui_url, process_info = ray._private.services.start_api_server(include_dashboard, raise_on_failure, self._ray_params.dashboard_host, self.gcs_address, self._node_ip_address, self._temp_dir, self._logs_dir, self._session_dir, port=self._ray_params.dashboard_port, dashboard_grpc_port=self._ray_params.dashboard_grpc_port, fate_share=self.kernel_fate_share, max_bytes=self.max_bytes, backup_count=self.backup_count, redirect_logging=self.should_redirect_logs(), stdout_file=stderr_file, stderr_file=stderr_file)
    assert ray_constants.PROCESS_TYPE_DASHBOARD not in self.all_processes
    if process_info is not None:
        self.all_processes[ray_constants.PROCESS_TYPE_DASHBOARD] = [process_info]
        self.get_gcs_client().internal_kv_put(b'webui:url', self._webui_url.encode(), True, ray_constants.KV_NAMESPACE_DASHBOARD)