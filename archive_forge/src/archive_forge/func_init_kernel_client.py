import atexit
import os
import signal
import sys
import typing as t
import uuid
import warnings
from jupyter_core.application import base_aliases, base_flags
from traitlets import CBool, CUnicode, Dict, List, Type, Unicode
from traitlets.config.application import boolean_flag
from . import KernelManager, connect, find_connection_file, tunnel_to_kernel
from .blocking import BlockingKernelClient
from .connect import KernelConnectionInfo
from .kernelspec import NoSuchKernel
from .localinterfaces import localhost
from .restarter import KernelRestarter
from .session import Session
from .utils import _filefind
def init_kernel_client(self) -> None:
    """Initialize the kernel client."""
    if self.kernel_manager is not None:
        self.kernel_client = self.kernel_manager.client()
    else:
        self.kernel_client = self.kernel_client_class(session=self.session, ip=self.ip, transport=self.transport, shell_port=self.shell_port, iopub_port=self.iopub_port, stdin_port=self.stdin_port, hb_port=self.hb_port, control_port=self.control_port, connection_file=self.connection_file, parent=self)
    self.kernel_client.start_channels()