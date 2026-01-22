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
def pre_start_kernel(self, kernel_name: str | None, kwargs: t.Any) -> tuple[KernelManager, str, str]:
    kernel_id = kwargs.pop('kernel_id', self.new_kernel_id(**kwargs))
    if kernel_id in self:
        raise DuplicateKernelError('Kernel already exists: %s' % kernel_id)
    if kernel_name is None:
        kernel_name = self.default_kernel_name
    constructor_kwargs = {}
    if self.kernel_spec_manager:
        constructor_kwargs['kernel_spec_manager'] = self.kernel_spec_manager
    km = self.kernel_manager_factory(connection_file=os.path.join(self.connection_dir, 'kernel-%s.json' % kernel_id), parent=self, log=self.log, kernel_name=kernel_name, **constructor_kwargs)
    return (km, kernel_name, kernel_id)