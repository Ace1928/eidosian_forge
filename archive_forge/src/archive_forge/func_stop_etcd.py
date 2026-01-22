import atexit
import logging
import os
import shlex
import shutil
import socket
import subprocess
import tempfile
import time
from typing import Optional, TextIO, Union
def stop_etcd(subprocess, data_dir: Optional[str]=None):
    if subprocess and subprocess.poll() is None:
        log.info('stopping etcd server')
        subprocess.terminate()
        subprocess.wait()
    if data_dir:
        log.info('deleting etcd data dir: %s', data_dir)
        shutil.rmtree(data_dir, ignore_errors=True)