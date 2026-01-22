import os.path
import subprocess
import sys
import time
import shutil
import fcntl
import signal
import socket
import logging
import threading
from ray.util.spark.cluster_init import (
from ray._private.ray_process_reaper import SIGTERM_GRACE_PERIOD_SECONDS
def check_parent_alive() -> None:
    orig_parent_pid = int(os.environ[RAY_ON_SPARK_START_RAY_PARENT_PID])
    while True:
        time.sleep(0.5)
        if os.getppid() != orig_parent_pid:
            process.terminate()
            try_clean_temp_dir_at_exit()
            os._exit(143)