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
def load_connection_file(self, connection_file: str | None=None) -> None:
    """Load connection info from JSON dict in self.connection_file.

        Parameters
        ----------
        connection_file: unicode, optional
            Path to connection file to load.
            If unspecified, use self.connection_file
        """
    if connection_file is None:
        connection_file = self.connection_file
    self.log.debug('Loading connection file %s', connection_file)
    with open(connection_file) as f:
        info = json.load(f)
    self.load_connection_info(info)