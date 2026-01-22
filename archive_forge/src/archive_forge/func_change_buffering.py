import asyncio
from collections import deque
import errno
import fcntl
import gc
import getpass
import glob as glob_module
import inspect
import logging
import os
import platform
import pty
import pwd
import re
import select
import signal
import stat
import struct
import sys
import termios
import textwrap
import threading
import time
import traceback
import tty
import warnings
import weakref
from asyncio import Queue as AQueue
from contextlib import contextmanager
from functools import partial
from importlib import metadata
from io import BytesIO, StringIO, UnsupportedOperation
from io import open as fdopen
from locale import getpreferredencoding
from queue import Empty, Queue
from shlex import quote as shlex_quote
from types import GeneratorType, ModuleType
from typing import Any, Dict, Type, Union
def change_buffering(self, new_type):
    self.log.debug('acquiring buffering lock for changing buffering')
    self._buffering_lock.acquire()
    self.log.debug('got buffering lock for changing buffering')
    try:
        if new_type == 0:
            self._use_up_buffer_first = True
        self.type = new_type
    finally:
        self._buffering_lock.release()
        self.log.debug('released buffering lock for changing buffering')