from __future__ import annotations
import asyncio
import json
import os
import socket
import typing as t
import uuid
from functools import wraps
from pathlib import Path
import zmq
from traitlets import Any, Bool, Dict, DottedObjectName, Instance, Unicode, default, observe
from traitlets.config.configurable import LoggingConfigurable
from traitlets.utils.importstring import import_item
from .connect import KernelConnectionInfo
from .kernelspec import NATIVE_KERNEL_NAME, KernelSpecManager
from .manager import KernelManager
from .utils import ensure_async, run_sync, utcnow
@kernel_method
def finish_shutdown(self, kernel_id: str, waittime: float | None=None, pollinterval: float | None=0.1) -> None:
    """Wait for a kernel to finish shutting down, and kill it if it doesn't"""
    self.log.info('Kernel shutdown: %s', kernel_id)