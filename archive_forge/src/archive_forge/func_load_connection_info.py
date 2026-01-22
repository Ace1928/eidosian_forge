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
def load_connection_info(self, info: KernelConnectionInfo) -> None:
    """Load connection info from a dict containing connection info.

        Typically this data comes from a connection file
        and is called by load_connection_file.

        Parameters
        ----------
        info: dict
            Dictionary containing connection_info.
            See the connection_file spec for details.
        """
    self.transport = info.get('transport', self.transport)
    self.ip = info.get('ip', self._ip_default())
    self._record_random_port_names()
    for name in port_names:
        if getattr(self, name) == 0 and name in info:
            setattr(self, name, info[name])
    if 'key' in info:
        key = info['key']
        if isinstance(key, str):
            key = key.encode()
        assert isinstance(key, bytes)
        self.session.key = key
    if 'signature_scheme' in info:
        self.session.signature_scheme = info['signature_scheme']