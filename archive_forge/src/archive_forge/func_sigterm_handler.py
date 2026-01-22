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
def sigterm_handler(*args):
    process.terminate()
    try_clean_temp_dir_at_exit()
    os._exit(143)