from __future__ import print_function
import os
import socket
import signal
import threading
from contextlib import closing, contextmanager
from . import _gi
@contextmanager
def wakeup_on_signal():
    """A decorator for functions which create a glib event loop to keep
    Python signal handlers working while the event loop is idling.

    In case an OS signal is received will wake the default event loop up
    shortly so that any registered Python signal handlers registered through
    signal.signal() can run.

    In case the wrapped function is not called from the main thread it will be
    called as is and it will not wake up the default loop for signals.
    """
    global _wakeup_fd_is_active
    if _wakeup_fd_is_active:
        yield
        return
    from gi.repository import GLib
    read_socket, write_socket = socket.socketpair()
    with closing(read_socket), closing(write_socket):
        for sock in [read_socket, write_socket]:
            sock.setblocking(False)
            ensure_socket_not_inheritable(sock)
        try:
            orig_fd = signal.set_wakeup_fd(write_socket.fileno())
        except ValueError:
            yield
            return
        else:
            _wakeup_fd_is_active = True

        def signal_notify(source, condition):
            if condition & GLib.IO_IN:
                try:
                    return bool(read_socket.recv(1))
                except EnvironmentError as e:
                    print(e)
                    return False
                return True
            else:
                return False
        try:
            if os.name == 'nt':
                channel = GLib.IOChannel.win32_new_socket(read_socket.fileno())
            else:
                channel = GLib.IOChannel.unix_new(read_socket.fileno())
            source_id = GLib.io_add_watch(channel, GLib.PRIORITY_DEFAULT, GLib.IOCondition.IN | GLib.IOCondition.HUP | GLib.IOCondition.NVAL | GLib.IOCondition.ERR, signal_notify)
            try:
                yield
            finally:
                GLib.source_remove(source_id)
        finally:
            write_fd = signal.set_wakeup_fd(orig_fd)
            if write_fd != write_socket.fileno():
                signal.set_wakeup_fd(write_fd)
            _wakeup_fd_is_active = False