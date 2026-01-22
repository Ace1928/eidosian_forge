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
def curve_public(secret_key) -> bytes:
    """Compute the public key corresponding to a secret key for use
    with zmq.CURVE security

    Requires libzmq (≥ 4.2) to have been built with CURVE support.

    Parameters
    ----------
    private
        The private key as a 40 byte z85-encoded bytestring

    Returns
    -------
    bytes
        The public key as a 40 byte z85-encoded bytestring
    """
    if isinstance(secret_key, str):
        secret_key = secret_key.encode('utf8')
    if not len(secret_key) == 40:
        raise ValueError('secret key must be a 40 byte z85 encoded string')
    rc: C.int
    public_key = declare(char[64])
    c_secret_key: pointer(char) = secret_key
    _check_version((4, 2), 'curve_public')
    rc = zmq_curve_public(public_key, c_secret_key)
    _check_rc(rc)
    return public_key[:40]