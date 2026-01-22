from __future__ import annotations
import time
import warnings
from threading import Event
from weakref import ref
import cython as C
from cython import (
from cython.cimports.cpython import (
from cython.cimports.libc.errno import EAGAIN, EINTR, ENAMETOOLONG, ENOENT, ENOTSOCK
from cython.cimports.libc.stdint import uint32_t
from cython.cimports.libc.stdio import fprintf
from cython.cimports.libc.stdio import stderr as cstderr
from cython.cimports.libc.stdlib import free, malloc
from cython.cimports.libc.string import memcpy
from cython.cimports.zmq.backend.cython._externs import (
from cython.cimports.zmq.backend.cython.libzmq import (
from cython.cimports.zmq.backend.cython.libzmq import zmq_errno as _zmq_errno
from cython.cimports.zmq.backend.cython.libzmq import zmq_poll as zmq_poll_c
from cython.cimports.zmq.utils.buffers import asbuffer_r
import zmq
from zmq.constants import SocketOption, _OptType
from zmq.error import InterruptedSystemCall, ZMQError, _check_version
def curve_keypair() -> tuple[bytes, bytes]:
    """generate a Z85 key pair for use with zmq.CURVE security

    Requires libzmq (≥ 4.0) to have been built with CURVE support.

    .. versionadded:: libzmq-4.0
    .. versionadded:: 14.0

    Returns
    -------
    public: bytes
        The public key as 40 byte z85-encoded bytestring.
    private: bytes
        The private key as 40 byte z85-encoded bytestring.
    """
    rc: C.int
    public_key = declare(char[64])
    secret_key = declare(char[64])
    _check_version((4, 0), 'curve_keypair')
    rc = zmq_curve_keypair(public_key, secret_key)
    _check_rc(rc)
    return (public_key, secret_key)