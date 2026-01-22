from ._cares import ffi as _ffi, lib as _lib
import _cffi_backend  # hint for bundler tools
from . import errno
from .utils import ascii_bytes, maybe_str, parse_name
from ._version import __version__
import collections.abc
import socket
import math
import functools
import sys
def getnameinfo(self, address, flags, callback):
    if not callable(callback):
        raise TypeError('a callable is required')
    if len(address) == 2:
        ip, port = address
        sa4 = _ffi.new('struct sockaddr_in*')
        if _lib.ares_inet_pton(socket.AF_INET, ascii_bytes(ip), _ffi.addressof(sa4.sin_addr)) != 1:
            raise ValueError('Invalid IPv4 address %r' % ip)
        sa4.sin_family = socket.AF_INET
        sa4.sin_port = socket.htons(port)
        sa = sa4
    elif len(address) == 4:
        ip, port, flowinfo, scope_id = address
        sa6 = _ffi.new('struct sockaddr_in6*')
        if _lib.ares_inet_pton(socket.AF_INET6, ascii_bytes(ip), _ffi.addressof(sa6.sin6_addr)) != 1:
            raise ValueError('Invalid IPv6 address %r' % ip)
        sa6.sin6_family = socket.AF_INET6
        sa6.sin6_port = socket.htons(port)
        sa6.sin6_flowinfo = socket.htonl(flowinfo)
        sa6.sin6_scope_id = scope_id
        sa = sa6
    else:
        raise ValueError('Invalid address argument')
    userdata = _ffi.new_handle(callback)
    _global_set.add(userdata)
    _lib.ares_getnameinfo(self._channel[0], _ffi.cast('struct sockaddr*', sa), _ffi.sizeof(sa[0]), flags, _lib._nameinfo_cb, userdata)