import asyncio
import base64
import binascii
import contextlib
import datetime
import enum
import functools
import inspect
import netrc
import os
import platform
import re
import sys
import time
import warnings
import weakref
from collections import namedtuple
from contextlib import suppress
from email.parser import HeaderParser
from email.utils import parsedate
from math import ceil
from pathlib import Path
from types import TracebackType
from typing import (
from urllib.parse import quote
from urllib.request import getproxies, proxy_bypass
import attr
from multidict import MultiDict, MultiDictProxy, MultiMapping
from yarl import URL
from . import hdrs
from .log import client_logger, internal_logger
def get_running_loop(loop: Optional[asyncio.AbstractEventLoop]=None) -> asyncio.AbstractEventLoop:
    if loop is None:
        loop = asyncio.get_event_loop()
    if not loop.is_running():
        warnings.warn('The object should be created within an async function', DeprecationWarning, stacklevel=3)
        if loop.get_debug():
            internal_logger.warning('The object should be created within an async function', stack_info=True)
    return loop