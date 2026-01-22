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
def is_event(self, *event):
    """Returns True if this message is an Event of one of the specified types."""
    if not isinstance(self, Event):
        return False
    return event == () or self.event in event