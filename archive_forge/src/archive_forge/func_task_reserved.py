import os
import platform
import shelve
import sys
import weakref
import zlib
from collections import Counter
from kombu.serialization import pickle, pickle_protocol
from kombu.utils.objects import cached_property
from celery import __version__
from celery.exceptions import WorkerShutdown, WorkerTerminate
from celery.utils.collections import LimitedSet
def task_reserved(request):
    """Called when a task is reserved by the worker."""
    global bench_start
    global bench_first
    now = None
    if bench_start is None:
        bench_start = now = monotonic()
    if bench_first is None:
        bench_first = now
    return __reserved(request)