import sys
import warnings
import os.path
import re
import subprocess
import threading
import pytest
import _pytest
def check_subprocess_call(cmd, timeout=5, stdout_regex=None, stderr_regex=None):
    """Runs a command in a subprocess with timeout in seconds.

    A SIGTERM is sent after `timeout` and if it does not terminate, a
    SIGKILL is sent after `2 * timeout`.

    Also checks returncode is zero, stdout if stdout_regex is set, and
    stderr if stderr_regex is set.
    """
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    def terminate_process():
        """
        Attempt to terminate a leftover process spawned during test execution:
        ideally this should not be needed but can help avoid clogging the CI
        workers in case of deadlocks.
        """
        warnings.warn(f'Timeout running {cmd}')
        proc.terminate()

    def kill_process():
        """
        Kill a leftover process spawned during test execution: ideally this
        should not be needed but can help avoid clogging the CI workers in
        case of deadlocks.
        """
        warnings.warn(f'Timeout running {cmd}')
        proc.kill()
    try:
        if timeout is not None:
            terminate_timer = threading.Timer(timeout, terminate_process)
            terminate_timer.start()
            kill_timer = threading.Timer(2 * timeout, kill_process)
            kill_timer.start()
        stdout, stderr = proc.communicate()
        stdout, stderr = (stdout.decode(), stderr.decode())
        if proc.returncode != 0:
            message = 'Non-zero return code: {}.\nStdout:\n{}\nStderr:\n{}'.format(proc.returncode, stdout, stderr)
            raise ValueError(message)
        if stdout_regex is not None and (not re.search(stdout_regex, stdout)):
            raise ValueError('Unexpected stdout: {!r} does not match:\n{!r}'.format(stdout_regex, stdout))
        if stderr_regex is not None and (not re.search(stderr_regex, stderr)):
            raise ValueError('Unexpected stderr: {!r} does not match:\n{!r}'.format(stderr_regex, stderr))
    finally:
        if timeout is not None:
            terminate_timer.cancel()
            kill_timer.cancel()