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
def dont_wait_for_first_connection():
    """Unblocks any pending wait_until_disconnected() call that is waiting on the
    first server to connect.
    """
    with _lock:
        _connections_changed.set()