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
def cant_handle(self, *args, **kwargs):
    """Same as self.error(MessageHandlingError, ...)."""
    return self.error(MessageHandlingError, *args, **kwargs)