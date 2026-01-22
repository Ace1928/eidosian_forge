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
def env_validator(passed_kwargs, merged_kwargs):
    """a validator to check that env is a dictionary and that all environment variable
    keys and values are strings. Otherwise, we would exit with a confusing exit code
    255."""
    invalid = []
    env = passed_kwargs.get('env', None)
    if env is None:
        return invalid
    if not isinstance(env, Mapping):
        invalid.append(('env', f'env must be dict-like. Got {env!r}'))
        return invalid
    for k, v in passed_kwargs['env'].items():
        if not isinstance(k, str):
            invalid.append(('env', f'env key {k!r} must be a str'))
        if not isinstance(v, str):
            invalid.append(('env', f'value {v!r} of env key {k!r} must be a str'))
    return invalid