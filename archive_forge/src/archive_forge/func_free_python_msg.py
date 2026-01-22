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
@cfunc
@nogil
def free_python_msg(data: p_void, vhint: p_void) -> C.int:
    """A pure-C function for DECREF'ing Python-owned message data.

    Sends a message on a PUSH socket

    The hint is a `zhint` struct with two values:

    sock (void *): pointer to the Garbage Collector's PUSH socket
    id (size_t): the id to be used to construct a zmq_msg_t that should be sent on a PUSH socket,
       signaling the Garbage Collector to remove its reference to the object.

    When the Garbage Collector's PULL socket receives the message,
    it deletes its reference to the object,
    allowing Python to free the memory.
    """
    msg = declare(zmq_msg_t)
    msg_ptr: pointer(zmq_msg_t) = address(msg)
    hint: pointer(_zhint) = cast(pointer(_zhint), vhint)
    rc: C.int
    if hint != NULL:
        zmq_msg_init_size(msg_ptr, sizeof(size_t))
        memcpy(zmq_msg_data(msg_ptr), address(hint.id), sizeof(size_t))
        rc = mutex_lock(hint.mutex)
        if rc != 0:
            fprintf(cstderr, 'pyzmq-gc mutex lock failed rc=%d\n', rc)
        rc = zmq_msg_send(msg_ptr, hint.sock, 0)
        if rc < 0:
            if _zmq_errno() != ZMQ_ENOTSOCK:
                fprintf(cstderr, 'pyzmq-gc send failed: %s\n', zmq_strerror(_zmq_errno()))
        rc = mutex_unlock(hint.mutex)
        if rc != 0:
            fprintf(cstderr, 'pyzmq-gc mutex unlock failed rc=%d\n', rc)
        zmq_msg_close(msg_ptr)
        free(hint)
        return 0