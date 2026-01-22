import functools
import logging
import multiprocessing
import os
import random
import shlex
import signal
import sys
import time
import warnings
import enum
from oslo_utils import encodeutils
from oslo_utils import importutils
from oslo_utils import strutils
from oslo_utils import timeutils
from oslo_concurrency._i18n import _
def ssh_execute(ssh, cmd, process_input=None, addl_env=None, check_exit_code=True, binary=False, timeout=None, sanitize_stdout=True):
    """Run a command through SSH.

    :param ssh:             An SSH Connection object.
    :param cmd:             The command string to run.
    :param check_exit_code: If an exception should be raised for non-zero
                            exit.
    :param timeout:         Max time in secs to wait for command execution.
    :param sanitize_stdout: Defaults to True. If set to True, stdout is
                            sanitized i.e. any sensitive information like
                            password in command output will be masked.
    :returns:               (stdout, stderr) from command execution through
                            SSH.

    .. versionchanged:: 1.9
       Added *binary* optional parameter.
    """
    sanitized_cmd = strutils.mask_password(cmd)
    LOG.debug('Running cmd (SSH): %s', sanitized_cmd)
    if addl_env:
        raise InvalidArgumentError(_('Environment not supported over SSH'))
    if process_input:
        raise InvalidArgumentError(_('process_input not supported over SSH'))
    stdin_stream, stdout_stream, stderr_stream = ssh.exec_command(cmd, timeout=timeout)
    channel = stdout_stream.channel
    stdout = stdout_stream.read()
    stderr = stderr_stream.read()
    stdin_stream.close()
    exit_status = channel.recv_exit_status()
    stdout = os.fsdecode(stdout)
    stderr = os.fsdecode(stderr)
    if sanitize_stdout:
        stdout = strutils.mask_password(stdout)
    stderr = strutils.mask_password(stderr)
    if exit_status != -1:
        LOG.debug('Result was %s' % exit_status)
        if check_exit_code and exit_status != 0:
            stdout = strutils.mask_password(stdout)
            raise ProcessExecutionError(exit_code=exit_status, stdout=stdout, stderr=stderr, cmd=sanitized_cmd)
    if binary:
        stdout = os.fsencode(stdout)
        stderr = os.fsencode(stderr)
    return (stdout, stderr)