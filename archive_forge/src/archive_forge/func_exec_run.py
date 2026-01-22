import copy
import ntpath
from collections import namedtuple
from ..api import APIClient
from ..constants import DEFAULT_DATA_CHUNK_SIZE
from ..errors import (
from ..types import HostConfig
from ..utils import version_gte
from .images import Image
from .resource import Collection, Model
def exec_run(self, cmd, stdout=True, stderr=True, stdin=False, tty=False, privileged=False, user='', detach=False, stream=False, socket=False, environment=None, workdir=None, demux=False):
    """
        Run a command inside this container. Similar to
        ``docker exec``.

        Args:
            cmd (str or list): Command to be executed
            stdout (bool): Attach to stdout. Default: ``True``
            stderr (bool): Attach to stderr. Default: ``True``
            stdin (bool): Attach to stdin. Default: ``False``
            tty (bool): Allocate a pseudo-TTY. Default: False
            privileged (bool): Run as privileged.
            user (str): User to execute command as. Default: root
            detach (bool): If true, detach from the exec command.
                Default: False
            stream (bool): Stream response data. Default: False
            socket (bool): Return the connection socket to allow custom
                read/write operations. Default: False
            environment (dict or list): A dictionary or a list of strings in
                the following format ``["PASSWORD=xxx"]`` or
                ``{"PASSWORD": "xxx"}``.
            workdir (str): Path to working directory for this exec session
            demux (bool): Return stdout and stderr separately

        Returns:
            (ExecResult): A tuple of (exit_code, output)
                exit_code: (int):
                    Exit code for the executed command or ``None`` if
                    either ``stream`` or ``socket`` is ``True``.
                output: (generator, bytes, or tuple):
                    If ``stream=True``, a generator yielding response chunks.
                    If ``socket=True``, a socket object for the connection.
                    If ``demux=True``, a tuple of two bytes: stdout and stderr.
                    A bytestring containing response data otherwise.

        Raises:
            :py:class:`docker.errors.APIError`
                If the server returns an error.
        """
    resp = self.client.api.exec_create(self.id, cmd, stdout=stdout, stderr=stderr, stdin=stdin, tty=tty, privileged=privileged, user=user, environment=environment, workdir=workdir)
    exec_output = self.client.api.exec_start(resp['Id'], detach=detach, tty=tty, stream=stream, socket=socket, demux=demux)
    if socket or stream:
        return ExecResult(None, exec_output)
    return ExecResult(self.client.api.exec_inspect(resp['Id'])['ExitCode'], exec_output)