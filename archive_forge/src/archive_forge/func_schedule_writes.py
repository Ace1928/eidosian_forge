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
def schedule_writes(ready_fds, total_write_count=None):
    if not total_write_count:
        total_write_count = [0]
    num_ready = len(ready_fds)
    for _ in range(num_ready):
        ready_fd = ready_fds[total_write_count[0] % num_ready]
        total_write_count[0] += 1
        if ready_fd in active_writes:
            continue
        if is_fair_strategy and ready_fd in busy_workers:
            continue
        if ready_fd not in all_inqueues:
            hub_remove(ready_fd)
            continue
        try:
            job = pop_message()
        except IndexError:
            for inqfd in diff(active_writes):
                hub_remove(inqfd)
            break
        else:
            if not job._accepted:
                try:
                    proc = job._scheduled_for = fileno_to_inq[ready_fd]
                except KeyError:
                    put_message(job)
                    continue
                cor = _write_job(proc, ready_fd, job)
                job._writer = ref(cor)
                mark_write_gen_as_active(cor)
                mark_write_fd_as_active(ready_fd)
                mark_worker_as_busy(ready_fd)
                try:
                    next(cor)
                except StopIteration:
                    pass
                except OSError as exc:
                    if exc.errno != errno.EBADF:
                        raise
                else:
                    add_writer(ready_fd, cor)