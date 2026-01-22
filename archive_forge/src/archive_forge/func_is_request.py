from __future__ import annotations
import collections
import contextlib
import functools
import itertools
import os
import socket
import sys
import threading
from debugpy.common import json, log, util
from debugpy.common.util import hide_thread_from_debugger
def is_request(self, *command):
    """Returns True if this message is a Request of one of the specified types."""
    if not isinstance(self, Request):
        return False
    return command == () or self.command in command