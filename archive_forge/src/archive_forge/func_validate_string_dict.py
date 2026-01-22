import asyncio
import inspect
import sys
import time
import typing as t
from functools import partial
from getpass import getpass
from queue import Empty
import zmq.asyncio
from jupyter_core.utils import ensure_async
from traitlets import Any, Bool, Instance, Type
from .channels import major_protocol_version
from .channelsabc import ChannelABC, HBChannelABC
from .clientabc import KernelClientABC
from .connect import ConnectionFileMixin
from .session import Session
def validate_string_dict(dct: t.Dict[str, str]) -> None:
    """Validate that the input is a dict with string keys and values.

    Raises ValueError if not."""
    for k, v in dct.items():
        if not isinstance(k, str):
            raise ValueError('key %r in dict must be a string' % k)
        if not isinstance(v, str):
            raise ValueError('value %r in dict must be a string' % v)