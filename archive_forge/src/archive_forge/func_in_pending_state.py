import asyncio
import functools
import os
import re
import signal
import sys
import typing as t
import uuid
import warnings
from asyncio.futures import Future
from concurrent.futures import Future as CFuture
from contextlib import contextmanager
from enum import Enum
import zmq
from jupyter_core.utils import run_sync
from traitlets import (
from traitlets.utils.importstring import import_item
from . import kernelspec
from .asynchronous import AsyncKernelClient
from .blocking import BlockingKernelClient
from .client import KernelClient
from .connect import ConnectionFileMixin
from .managerabc import KernelManagerABC
from .provisioning import KernelProvisionerBase
from .provisioning import KernelProvisionerFactory as KPF  # noqa
def in_pending_state(method: F) -> F:
    """Sets the kernel to a pending state by
    creating a fresh Future for the KernelManager's `ready`
    attribute. Once the method is finished, set the Future's results.
    """

    @t.no_type_check
    @functools.wraps(method)
    async def wrapper(self: t.Any, *args: t.Any, **kwargs: t.Any) -> t.Any:
        """Create a future for the decorated method."""
        if self._attempted_start or not self._ready:
            self._ready = _get_future()
        try:
            out = await method(self, *args, **kwargs)
            await asyncio.sleep(0.01)
            if self.owns_kernel:
                self._ready.set_result(None)
            return out
        except Exception as e:
            self._ready.set_exception(e)
            self.log.exception(self._ready.exception())
            raise e
    return t.cast(F, wrapper)