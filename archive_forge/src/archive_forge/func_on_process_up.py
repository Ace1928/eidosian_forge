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
def on_process_up(proc):
    """Called when a process has started."""
    infd = proc.inqW_fd
    for job in cache.values():
        if job._write_to and job._write_to.inqW_fd == infd:
            job._write_to = proc
        if job._scheduled_for and job._scheduled_for.inqW_fd == infd:
            job._scheduled_for = proc
    fileno_to_outq[proc.outqR_fd] = proc
    self._track_child_process(proc, hub)
    assert not isblocking(proc.outq._reader)
    add_reader(proc.outqR_fd, handle_result_event, proc.outqR_fd)
    waiting_to_start.add(proc)
    hub.call_later(self._proc_alive_timeout, verify_process_alive, ref(proc))