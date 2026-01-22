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
def zmq_poll(sockets, timeout: C.int=-1):
    """zmq_poll(sockets, timeout=-1)

    Poll a set of 0MQ sockets, native file descs. or sockets.

    Parameters
    ----------
    sockets : list of tuples of (socket, flags)
        Each element of this list is a two-tuple containing a socket
        and a flags. The socket may be a 0MQ socket or any object with
        a ``fileno()`` method. The flags can be zmq.POLLIN (for detecting
        for incoming messages), zmq.POLLOUT (for detecting that send is OK)
        or zmq.POLLIN|zmq.POLLOUT for detecting both.
    timeout : int
        The number of milliseconds to poll for. Negative means no timeout.
    """
    rc: C.int
    i: C.int
    pollitems: pointer(zmq_pollitem_t) = NULL
    nsockets: C.int = len(sockets)
    if nsockets == 0:
        return []
    pollitems = cast(pointer(zmq_pollitem_t), malloc(nsockets * sizeof(zmq_pollitem_t)))
    if pollitems == NULL:
        raise MemoryError('Could not allocate poll items')
    if ZMQ_VERSION_MAJOR < 3:
        timeout = 1000 * timeout
    for i in range(nsockets):
        s, events = sockets[i]
        if isinstance(s, Socket):
            pollitems[i].socket = cast(Socket, s).handle
            pollitems[i].fd = 0
            pollitems[i].events = events
            pollitems[i].revents = 0
        elif isinstance(s, int):
            pollitems[i].socket = NULL
            pollitems[i].fd = s
            pollitems[i].events = events
            pollitems[i].revents = 0
        elif hasattr(s, 'fileno'):
            try:
                fileno = int(s.fileno())
            except Exception:
                free(pollitems)
                raise ValueError('fileno() must return a valid integer fd')
            else:
                pollitems[i].socket = NULL
                pollitems[i].fd = fileno
                pollitems[i].events = events
                pollitems[i].revents = 0
        else:
            free(pollitems)
            raise TypeError('Socket must be a 0MQ socket, an integer fd or have a fileno() method: %r' % s)
    ms_passed: int = 0
    try:
        while True:
            start = time.monotonic()
            with nogil:
                rc = zmq_poll_c(pollitems, nsockets, timeout)
            try:
                _check_rc(rc)
            except InterruptedSystemCall:
                if timeout > 0:
                    ms_passed = int(1000 * (time.monotonic() - start))
                    if ms_passed < 0:
                        warnings.warn(f'Negative elapsed time for interrupted poll: {ms_passed}.  Did the clock change?', RuntimeWarning)
                        ms_passed = 0
                    timeout = max(0, timeout - ms_passed)
                continue
            else:
                break
    except Exception:
        free(pollitems)
        raise
    results = []
    for i in range(nsockets):
        revents = pollitems[i].revents
        if revents > 0:
            if pollitems[i].socket != NULL:
                s = sockets[i][0]
            else:
                s = pollitems[i].fd
            results.append((s, revents))
    free(pollitems)
    return results