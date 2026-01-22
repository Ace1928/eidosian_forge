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
def on_partial_read(self, job, proc):
    """Called when a job was partially written to exited child."""
    if not job._accepted:
        self._put_back(job)
    writer = _get_job_writer(job)
    if writer:
        self._active_writers.discard(writer)
        del writer
    if not proc.dead:
        proc.dead = True
        before = len(self._queues)
        try:
            queues = self._find_worker_queues(proc)
            if self.destroy_queues(queues, proc):
                self._queues[self.create_process_queues()] = None
        except ValueError:
            pass
        assert len(self._queues) == before