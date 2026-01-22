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
def format_kernel_cmd(self, extra_arguments: t.Optional[t.List[str]]=None) -> t.List[str]:
    """Replace templated args (e.g. {connection_file})"""
    extra_arguments = extra_arguments or []
    assert self.kernel_spec is not None
    cmd = self.kernel_spec.argv + extra_arguments
    if cmd and cmd[0] in {'python', 'python%i' % sys.version_info[0], 'python%i.%i' % sys.version_info[:2]}:
        cmd[0] = sys.executable
    ns: t.Dict[str, t.Any] = {'connection_file': os.path.realpath(self.connection_file), 'prefix': sys.prefix}
    if self.kernel_spec:
        ns['resource_dir'] = self.kernel_spec.resource_dir
    assert isinstance(self._launch_args, dict)
    ns.update(self._launch_args)
    pat = re.compile('\\{([A-Za-z0-9_]+)\\}')

    def from_ns(match: t.Any) -> t.Any:
        """Get the key out of ns if it's there, otherwise no change."""
        return ns.get(match.group(1), match.group())
    return [pat.sub(from_ns, arg) for arg in cmd]