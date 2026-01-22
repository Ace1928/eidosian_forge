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
def cleanup_random_ports(self) -> None:
    """Forgets randomly assigned port numbers and cleans up the connection file.

        Does nothing if no port numbers have been randomly assigned.
        In particular, does nothing unless the transport is tcp.
        """
    if not self._random_port_names:
        return
    for name in self._random_port_names:
        setattr(self, name, 0)
    self.cleanup_connection_file()