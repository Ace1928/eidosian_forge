import errno
import os
import signal
import sys
import time
import ovs.dirs
import ovs.fatal_signal
import ovs.process
import ovs.socket_util
import ovs.timeval
import ovs.util
import ovs.vlog
def daemonize_start():
    """If daemonization is configured, then starts daemonization, by forking
    and returning in the child process.  The parent process hangs around until
    the child lets it know either that it completed startup successfully (by
    calling daemon_complete()) or that it failed to start up (by exiting with a
    nonzero exit code)."""
    if _detach:
        if _fork_and_wait_for_startup() > 0:
            sys.exit(0)
        if sys.platform != 'win32':
            os.setsid()
    if _monitor:
        saved_daemonize_fd = _daemonize_fd
        daemon_pid = _fork_and_wait_for_startup()
        if daemon_pid > 0:
            _fork_notify_startup(saved_daemonize_fd)
            if sys.platform != 'win32':
                _close_standard_fds()
            _monitor_daemon(daemon_pid)
    if _pidfile:
        _make_pidfile()