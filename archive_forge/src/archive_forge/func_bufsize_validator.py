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
def bufsize_validator(passed_kwargs, merged_kwargs):
    """a validator to prevent a user from saying that they want custom
    buffering when they're using an in/out object that will be os.dup'ed to the
    process, and has its own buffering.  an example is a pipe or a tty.  it
    doesn't make sense to tell them to have a custom buffering, since the os
    controls this."""
    invalid = []
    in_ob = passed_kwargs.get('in', None)
    out_ob = passed_kwargs.get('out', None)
    in_buf = passed_kwargs.get('in_bufsize', None)
    out_buf = passed_kwargs.get('out_bufsize', None)
    in_no_buf = ob_is_fd_based(in_ob)
    out_no_buf = ob_is_fd_based(out_ob)
    err = "Can't specify an {target} bufsize if the {target} target is a pipe or TTY"
    if in_no_buf and in_buf is not None:
        invalid.append((('in', 'in_bufsize'), err.format(target='in')))
    if out_no_buf and out_buf is not None:
        invalid.append((('out', 'out_bufsize'), err.format(target='out')))
    return invalid