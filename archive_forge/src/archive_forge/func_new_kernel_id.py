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
def new_kernel_id(self, **kwargs: t.Any) -> str:
    """
        Returns the id to associate with the kernel for this request. Subclasses may override
        this method to substitute other sources of kernel ids.
        :param kwargs:
        :return: string-ized version 4 uuid
        """
    return str(uuid.uuid4())