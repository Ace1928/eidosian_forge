import errno
import gc
import inspect
import os
import select
import time
from collections import Counter, deque, namedtuple
from io import BytesIO
from numbers import Integral
from pickle import HIGHEST_PROTOCOL
from struct import pack, unpack, unpack_from
from time import sleep
from weakref import WeakValueDictionary, ref
from billiard import pool as _pool
from billiard.compat import isblocking, setblocking
from billiard.pool import ACK, NACK, RUN, TERMINATE, WorkersJoined
from billiard.queues import _SimpleQueue
from kombu.asynchronous import ERR, WRITE
from kombu.serialization import pickle as _pickle
from kombu.utils.eventio import SELECT_BAD_FD
from kombu.utils.functional import fxrange
from vine import promise
from celery.signals import worker_before_create_process
from celery.utils.functional import noop
from celery.utils.log import get_logger
from celery.worker import state as worker_state
def create_process_queues(self):
    """Create new in, out, etc. queues, returned as a tuple."""
    inq = _SimpleQueue(wnonblock=True)
    outq = _SimpleQueue(rnonblock=True)
    synq = None
    assert isblocking(inq._reader)
    assert not isblocking(inq._writer)
    assert not isblocking(outq._reader)
    assert isblocking(outq._writer)
    if self.synack:
        synq = _SimpleQueue(wnonblock=True)
        assert isblocking(synq._reader)
        assert not isblocking(synq._writer)
    return (inq, outq, synq)