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
def write_connection_file(self, **kwargs: Any) -> None:
    """Write connection info to JSON dict in self.connection_file."""
    if self._connection_file_written and os.path.exists(self.connection_file):
        return
    self.connection_file, cfg = write_connection_file(self.connection_file, transport=self.transport, ip=self.ip, key=self.session.key, stdin_port=self.stdin_port, iopub_port=self.iopub_port, shell_port=self.shell_port, hb_port=self.hb_port, control_port=self.control_port, signature_scheme=self.session.signature_scheme, kernel_name=self.kernel_name, **kwargs)
    self._record_random_port_names()
    for name in port_names:
        setattr(self, name, cfg[name])
    self._connection_file_written = True