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
def basicauth_from_netrc(netrc_obj: Optional[netrc.netrc], host: str) -> BasicAuth:
    """
    Return :py:class:`~aiohttp.BasicAuth` credentials for ``host`` from ``netrc_obj``.

    :raises LookupError: if ``netrc_obj`` is :py:data:`None` or if no
            entry is found for the ``host``.
    """
    if netrc_obj is None:
        raise LookupError('No .netrc file found')
    auth_from_netrc = netrc_obj.authenticators(host)
    if auth_from_netrc is None:
        raise LookupError(f'No entry for {host!s} found in the `.netrc` file.')
    login, account, password = auth_from_netrc
    username = login if login or account is None else account
    if password is None:
        password = ''
    return BasicAuth(username, password)