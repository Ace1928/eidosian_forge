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
def on_stop_not_started(self):
    cache = self.cache
    check_timeouts = self.check_timeouts
    fileno_to_outq = self.fileno_to_outq
    on_state_change = self.on_state_change
    join_exited_workers = self.join_exited_workers
    outqueues = set(fileno_to_outq)
    while cache and outqueues and (self._state != TERMINATE):
        if check_timeouts is not None:
            check_timeouts()
        pending_remove_fd = set()
        for fd in outqueues:
            iterate_file_descriptors_safely([fd], self.fileno_to_outq, self._flush_outqueue, pending_remove_fd.add, fileno_to_outq, on_state_change)
            try:
                join_exited_workers(shutdown=True)
            except WorkersJoined:
                debug('result handler: all workers terminated')
                return
        outqueues.difference_update(pending_remove_fd)