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
def pack_command(self, *args: EncodableT) -> List[bytes]:
    """Pack a series of arguments into the Redis protocol"""
    output = []
    assert not isinstance(args[0], float)
    if isinstance(args[0], str):
        args = tuple(args[0].encode().split()) + args[1:]
    elif b' ' in args[0]:
        args = tuple(args[0].split()) + args[1:]
    buff = SYM_EMPTY.join((SYM_STAR, str(len(args)).encode(), SYM_CRLF))
    buffer_cutoff = self._buffer_cutoff
    for arg in map(self.encoder.encode, args):
        arg_length = len(arg)
        if len(buff) > buffer_cutoff or arg_length > buffer_cutoff or isinstance(arg, memoryview):
            buff = SYM_EMPTY.join((buff, SYM_DOLLAR, str(arg_length).encode(), SYM_CRLF))
            output.append(buff)
            output.append(arg)
            buff = SYM_CRLF
        else:
            buff = SYM_EMPTY.join((buff, SYM_DOLLAR, str(arg_length).encode(), SYM_CRLF, arg, SYM_CRLF))
    output.append(buff)
    return output