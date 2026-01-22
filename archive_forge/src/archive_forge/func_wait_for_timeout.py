from __future__ import annotations
import os
import subprocess
import sys
import threading
import time
import debugpy
from debugpy import adapter
from debugpy.common import json, log, messaging, sockets
from debugpy.adapter import components, sessions
import traceback
import io
def wait_for_timeout():
    time.sleep(timeout)
    wait_for_timeout.timed_out = True
    with _lock:
        _connections_changed.set()