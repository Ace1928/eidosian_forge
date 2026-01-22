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
def cleanup_connection_file(self) -> None:
    """Cleanup connection file *if we wrote it*

        Will not raise if the connection file was already removed somehow.
        """
    if self._connection_file_written:
        self._connection_file_written = False
        try:
            os.remove(self.connection_file)
        except (OSError, AttributeError):
            pass