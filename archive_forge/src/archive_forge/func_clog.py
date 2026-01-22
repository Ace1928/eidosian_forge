from __future__ import annotations
import signal
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from multiprocessing.pool import ThreadPool
from time import sleep, time
import pytest
import dask
from dask.system import CPU_COUNT
from dask.threaded import get
from dask.utils_test import add, inc
def clog(in_clog_event: threading.Event, clog_event: threading.Event) -> None:
    in_clog_event.set()
    clog_event.wait()