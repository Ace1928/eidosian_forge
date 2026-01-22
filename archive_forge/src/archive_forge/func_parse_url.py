import asyncio
import copy
import enum
import inspect
import socket
import ssl
import sys
import warnings
import weakref
from abc import abstractmethod
from itertools import chain
from types import MappingProxyType
from typing import (
from urllib.parse import ParseResult, parse_qs, unquote, urlparse
from redis.asyncio.retry import Retry
from redis.backoff import NoBackoff
from redis.compat import Protocol, TypedDict
from redis.connection import DEFAULT_RESP_VERSION
from redis.credentials import CredentialProvider, UsernamePasswordCredentialProvider
from redis.exceptions import (
from redis.typing import EncodableT
from redis.utils import HIREDIS_AVAILABLE, get_lib_version, str_if_bytes
from .._parsers import (
def parse_url(url: str) -> ConnectKwargs:
    parsed: ParseResult = urlparse(url)
    kwargs: ConnectKwargs = {}
    for name, value_list in parse_qs(parsed.query).items():
        if value_list and len(value_list) > 0:
            value = unquote(value_list[0])
            parser = URL_QUERY_ARGUMENT_PARSERS.get(name)
            if parser:
                try:
                    kwargs[name] = parser(value)
                except (TypeError, ValueError):
                    raise ValueError(f'Invalid value for `{name}` in connection URL.')
            else:
                kwargs[name] = value
    if parsed.username:
        kwargs['username'] = unquote(parsed.username)
    if parsed.password:
        kwargs['password'] = unquote(parsed.password)
    if parsed.scheme == 'unix':
        if parsed.path:
            kwargs['path'] = unquote(parsed.path)
        kwargs['connection_class'] = UnixDomainSocketConnection
    elif parsed.scheme in ('redis', 'rediss'):
        if parsed.hostname:
            kwargs['host'] = unquote(parsed.hostname)
        if parsed.port:
            kwargs['port'] = int(parsed.port)
        if parsed.path and 'db' not in kwargs:
            try:
                kwargs['db'] = int(unquote(parsed.path).replace('/', ''))
            except (AttributeError, ValueError):
                pass
        if parsed.scheme == 'rediss':
            kwargs['connection_class'] = SSLConnection
    else:
        valid_schemes = 'redis://, rediss://, unix://'
        raise ValueError(f'Redis URL must specify one of the following schemes ({valid_schemes})')
    return kwargs