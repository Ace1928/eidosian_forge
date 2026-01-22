import os
import sys
import signal
import threading
import logging
from dataclasses import dataclass
from functools import partialmethod
from typing import Optional, Any, Dict, List
from .mp_utils import _CPU_CORES, _MAX_THREADS, _MAX_PROCS
@classmethod
def set_multiparams(cls, max_procs: int=None, max_threads: int=None):
    if max_procs is not None:
        EnvChecker.max_procs = max_procs
    if max_threads is not None:
        EnvChecker.max_threads = max_threads