from __future__ import annotations
import contextlib
import dataclasses
import functools
import logging
import os
import queue
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from ctypes import byref, c_size_t, c_void_p
from multiprocessing.process import BaseProcess
from multiprocessing.queues import Queue
from typing import (
import torch
from torch import multiprocessing
from torch._dynamo.testing import rand_strided
from torch._inductor import ir
from torch._inductor.codecache import CUDACodeCache, DLLWrapper, PyCodeCache
from . import config
from .utils import do_bench
from .virtualized import V
@staticmethod
def process_main(request_queue: Queue[Any], response_queue: Queue[Any]) -> None:
    """
        Entry point for the child process.
        """
    log.debug('Entering TuningProcess child. Visible devices = %s', os.environ.get(CUDA_VISIBLE_DEVICES))
    try:
        TuningProcess.workloop(request_queue, response_queue)
    except Exception as ex:
        log.exception('Exception in TuningProcess: %s', ex)