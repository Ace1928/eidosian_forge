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
def detach_from_session(self):
    with _lock:
        self.is_connected = False
        self.channel.handlers = self.connection
        self.channel.name = self.channel.stream.name = str(self.connection)
        self.connection.server = None