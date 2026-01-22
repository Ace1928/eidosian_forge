import asyncio
import atexit
import base64
import collections
import datetime
import re
import signal
import typing as t
from contextlib import asynccontextmanager, contextmanager
from queue import Empty
from textwrap import dedent
from time import monotonic
from jupyter_client import KernelManager
from jupyter_client.client import KernelClient
from nbformat import NotebookNode
from nbformat.v4 import output_from_msg
from traitlets import Any, Bool, Callable, Dict, Enum, Integer, List, Type, Unicode, default
from traitlets.config.configurable import LoggingConfigurable
from .exceptions import (
from .output_widget import OutputWidget
from .util import ensure_async, run_hook, run_sync
@contextmanager
def setup_kernel(self, **kwargs: t.Any) -> t.Generator:
    """
        Context manager for setting up the kernel to execute a notebook.

        The assigns the Kernel Manager (``self.km``) if missing and Kernel Client(``self.kc``).

        When control returns from the yield it stops the client's zmq channels, and shuts
        down the kernel.
        """
    cleanup_kc = kwargs.pop('cleanup_kc', self.owns_km)
    if self.km is None:
        self.km = self.create_kernel_manager()
    if not self.km.has_kernel:
        self.start_new_kernel(**kwargs)
    if self.kc is None:
        self.start_new_kernel_client()
    try:
        yield
    finally:
        if cleanup_kc:
            self._cleanup_kernel()