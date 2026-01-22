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
def construct_streamreader_callback(process, handler):
    """here we're constructing a closure for our streamreader callback.  this
    is used in the case that we pass a callback into _out or _err, meaning we
    want to our callback to handle each bit of output

    we construct the closure based on how many arguments it takes.  the reason
    for this is to make it as easy as possible for people to use, without
    limiting them.  a new user will assume the callback takes 1 argument (the
    data).  as they get more advanced, they may want to terminate the process,
    or pass some stdin back, and will realize that they can pass a callback of
    more args"""
    implied_arg = 0
    partial_args = 0
    handler_to_inspect = handler
    if isinstance(handler, partial):
        partial_args = len(handler.args)
        handler_to_inspect = handler.func
    if inspect.ismethod(handler_to_inspect):
        implied_arg = 1
        num_args = get_num_args(handler_to_inspect)
    elif inspect.isfunction(handler_to_inspect):
        num_args = get_num_args(handler_to_inspect)
    else:
        implied_arg = 1
        num_args = get_num_args(handler_to_inspect.__call__)
    net_args = num_args - implied_arg - partial_args
    handler_args = ()
    if net_args == 1:
        handler_args = ()
    if net_args == 2:
        handler_args = (process.stdin,)
    elif net_args == 3:
        handler_args = (process.stdin, weakref.ref(process))

    def fn(chunk):
        a = handler_args
        if len(a) == 2:
            a = (handler_args[0], handler_args[1]())
        return handler(chunk, *a)
    return fn