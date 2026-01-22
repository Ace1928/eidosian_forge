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
@message_handler
def continued_event(self, event):
    if self.client.client_id not in ('visualstudio', 'vsformac'):
        self.client.propagate_after_start(event)