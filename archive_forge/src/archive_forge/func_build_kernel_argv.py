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
def build_kernel_argv(self, argv: object=None) -> None:
    """build argv to be passed to kernel subprocess

        Override in subclasses if any args should be passed to the kernel
        """
    self.kernel_argv = self.extra_args