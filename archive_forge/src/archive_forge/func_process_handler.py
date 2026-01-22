import subprocess
import shlex
import sys
import os
from IPython.utils import py3compat
def process_handler(cmd, callback, stderr=subprocess.PIPE):
    """Open a command in a shell subprocess and execute a callback.

    This function provides common scaffolding for creating subprocess.Popen()
    calls.  It creates a Popen object and then calls the callback with it.

    Parameters
    ----------
    cmd : str or list
        A command to be executed by the system, using :class:`subprocess.Popen`.
        If a string is passed, it will be run in the system shell. If a list is
        passed, it will be used directly as arguments.
    callback : callable
        A one-argument function that will be called with the Popen object.
    stderr : file descriptor number, optional
        By default this is set to ``subprocess.PIPE``, but you can also pass the
        value ``subprocess.STDOUT`` to force the subprocess' stderr to go into
        the same file descriptor as its stdout.  This is useful to read stdout
        and stderr combined in the order they are generated.

    Returns
    -------
    The return value of the provided callback is returned.
    """
    sys.stdout.flush()
    sys.stderr.flush()
    close_fds = sys.platform != 'win32'
    shell = isinstance(cmd, str)
    executable = None
    if shell and os.name == 'posix' and ('SHELL' in os.environ):
        executable = os.environ['SHELL']
    p = subprocess.Popen(cmd, shell=shell, executable=executable, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=stderr, close_fds=close_fds)
    try:
        out = callback(p)
    except KeyboardInterrupt:
        print('^C')
        sys.stdout.flush()
        sys.stderr.flush()
        out = None
    finally:
        if p.returncode is None:
            try:
                p.terminate()
                p.poll()
            except OSError:
                pass
        if p.returncode is None:
            try:
                p.kill()
            except OSError:
                pass
    return out