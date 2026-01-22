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
@classmethod
def from_stdio(cls, name='stdio'):
    """Creates a new instance that receives messages from sys.stdin, and sends
        them to sys.stdout.
        """
    return cls(sys.stdin.buffer, sys.stdout.buffer, name)