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
def tty_in_validator(passed_kwargs, merged_kwargs):
    pairs = (('tty_in', 'in'), ('tty_out', 'out'))
    invalid = []
    for tty_type, std in pairs:
        if tty_type in passed_kwargs and ob_is_tty(passed_kwargs.get(std, None)):
            error = f"`_{std}` is a TTY already, so so it doesn't make sense to set up a TTY with `_{tty_type}`"
            invalid.append(((tty_type, std), error))
    if merged_kwargs['unify_ttys'] and (not (merged_kwargs['tty_in'] and merged_kwargs['tty_out'])):
        invalid.append((('unify_ttys', 'tty_in', 'tty_out'), '`_tty_in` and `_tty_out` must both be True if `_unify_ttys` is True'))
    return invalid