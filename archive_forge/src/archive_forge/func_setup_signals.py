import os
import signal
import typing as t
import uuid
from jupyter_core.application import JupyterApp, base_flags
from tornado.ioloop import IOLoop
from traitlets import Unicode
from . import __version__
from .kernelspec import NATIVE_KERNEL_NAME, KernelSpecManager
from .manager import KernelManager
def setup_signals(self) -> None:
    """Shutdown on SIGTERM or SIGINT (Ctrl-C)"""
    if os.name == 'nt':
        return

    def shutdown_handler(signo: int, frame: t.Any) -> None:
        self.loop.add_callback_from_signal(self.shutdown, signo)
    for sig in [signal.SIGTERM, signal.SIGINT]:
        signal.signal(sig, shutdown_handler)