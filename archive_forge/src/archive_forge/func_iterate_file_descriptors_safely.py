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
def iterate_file_descriptors_safely(fds_iter, source_data, hub_method, *args, **kwargs):
    """Apply hub method to fds in iter, remove from list if failure.

    Some file descriptors may become stale through OS reasons
    or possibly other reasons, so safely manage our lists of FDs.
    :param fds_iter: the file descriptors to iterate and apply hub_method
    :param source_data: data source to remove FD if it renders OSError
    :param hub_method: the method to call with each fd and kwargs
    :*args to pass through to the hub_method;
    with a special syntax string '*fd*' represents a substitution
    for the current fd object in the iteration (for some callers).
    :**kwargs to pass through to the hub method (no substitutions needed)
    """

    def _meta_fd_argument_maker():
        call_args = args
        if '*fd*' in call_args:
            call_args = [fd if arg == '*fd*' else arg for arg in args]
        return call_args
    stale_fds = []
    for fd in fds_iter:
        hub_args, hub_kwargs = (_meta_fd_argument_maker(), kwargs)
        try:
            hub_method(fd, *hub_args, **hub_kwargs)
        except (OSError, FileNotFoundError):
            logger.warning('Encountered OSError when accessing fd %s ', fd, exc_info=True)
            stale_fds.append(fd)
    if source_data:
        for fd in stale_fds:
            try:
                if hasattr(source_data, 'remove'):
                    source_data.remove(fd)
                else:
                    source_data.pop(fd, None)
            except ValueError:
                logger.warning('ValueError trying to invalidate %s from %s', fd, source_data)