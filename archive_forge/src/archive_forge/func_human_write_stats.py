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
def human_write_stats(self):
    if self.write_stats is None:
        return 'N/A'
    vals = list(self.write_stats.values())
    total = sum(vals)

    def per(v, total):
        return f'{(float(v) / total if v else 0):.2f}'
    return {'total': total, 'avg': per(total / len(self.write_stats) if total else 0, total), 'all': ', '.join((per(v, total) for v in vals)), 'raw': ', '.join(map(str, vals)), 'strategy': SCHED_STRATEGY_TO_NAME.get(self.sched_strategy, self.sched_strategy), 'inqueues': {'total': len(self._all_inqueues), 'active': len(self._active_writes)}}