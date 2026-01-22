from __future__ import annotations
import contextlib
import signal
import socket
import warnings
from .. import _core
from .._util import is_main_thread
def wakeup_on_signals(self) -> None:
    assert self.old_wakeup_fd is None
    if not is_main_thread():
        return
    fd = self.write_sock.fileno()
    self.old_wakeup_fd = signal.set_wakeup_fd(fd, warn_on_full_buffer=False)
    if self.old_wakeup_fd != -1:
        warnings.warn(RuntimeWarning("It looks like Trio's signal handling code might have collided with another library you're using. If you're running Trio in guest mode, then this might mean you should set host_uses_signal_set_wakeup_fd=True. Otherwise, file a bug on Trio and we'll help you figure out what's going on."), stacklevel=1)