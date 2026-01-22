from __future__ import annotations
import os
import signal
import contextlib
import multiprocessing
from typing import Optional, List, TypeVar, Callable, Dict, Any, Union, TYPE_CHECKING
from lazyops.utils.lazy import lazy_import
from lazyops.imports._psutil import _psutil_available
def exit_gracefully(self, signum, frame):
    self.logger.warning(f'[{self.pid}] Received {self.signals[signum]} signal. Exiting gracefully')
    self.kill_now = True