from __future__ import annotations
import errno
import glob
import json
import os
import socket
import stat
import tempfile
import warnings
from getpass import getpass
from typing import TYPE_CHECKING, Any, Dict, Union, cast
import zmq
from jupyter_core.paths import jupyter_data_dir, jupyter_runtime_dir, secure_write
from traitlets import Bool, CaselessStrEnum, Instance, Integer, Type, Unicode, observe
from traitlets.config import LoggingConfigurable, SingletonConfigurable
from .localinterfaces import localhost
from .utils import _filefind
def tunnel_to_kernel(connection_info: str | KernelConnectionInfo, sshserver: str, sshkey: str | None=None) -> tuple[Any, ...]:
    """tunnel connections to a kernel via ssh

    This will open five SSH tunnels from localhost on this machine to the
    ports associated with the kernel.  They can be either direct
    localhost-localhost tunnels, or if an intermediate server is necessary,
    the kernel must be listening on a public IP.

    Parameters
    ----------
    connection_info : dict or str (path)
        Either a connection dict, or the path to a JSON connection file
    sshserver : str
        The ssh sever to use to tunnel to the kernel. Can be a full
        `user@server:port` string. ssh config aliases are respected.
    sshkey : str [optional]
        Path to file containing ssh key to use for authentication.
        Only necessary if your ssh config does not already associate
        a keyfile with the host.

    Returns
    -------

    (shell, iopub, stdin, hb, control) : ints
        The five ports on localhost that have been forwarded to the kernel.
    """
    from .ssh import tunnel
    if isinstance(connection_info, str):
        with open(connection_info) as f:
            connection_info = json.loads(f.read())
    cf = cast(Dict[str, Any], connection_info)
    lports = tunnel.select_random_ports(5)
    rports = (cf['shell_port'], cf['iopub_port'], cf['stdin_port'], cf['hb_port'], cf['control_port'])
    remote_ip = cf['ip']
    if tunnel.try_passwordless_ssh(sshserver, sshkey):
        password: bool | str = False
    else:
        password = getpass('SSH Password for %s: ' % sshserver)
    for lp, rp in zip(lports, rports):
        tunnel.ssh_tunnel(lp, rp, sshserver, remote_ip, sshkey, password)
    return tuple(lports)