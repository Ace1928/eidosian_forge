import os
import asyncio
from types import coroutine
from fileio import PathIO
from io import (
from functools import partial, singledispatch, wraps
from typing import TypeVar, Union
from aiofiles.threadpool.binary import AsyncBufferedIOBase, AsyncBufferedReader, AsyncFileIO
from aiofiles.threadpool.text import AsyncTextIOWrapper
from aiofiles.base import AiofilesContextManager
from tensorflow.python.platform.gfile import GFile
from tensorflow.python.lib.io import file_io as tfio
from tensorflow.python.lib.io import _pywrap_file_io
def async_open(file, mode='r', buffering=-1, encoding=None, errors=None, newline=None, closefd=True, opener=None, *, loop=None, executor=None):
    return AiofilesContextManager(_open(file, mode=mode, buffering=buffering, encoding=encoding, errors=errors, newline=newline, closefd=closefd, opener=opener, loop=loop, executor=executor))