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
def blocking_client(self) -> BlockingKernelClient:
    """Make a blocking client connected to my kernel"""
    info = self.get_connection_info()
    bc = self.blocking_class(parent=self)
    bc.load_connection_info(info)
    return bc