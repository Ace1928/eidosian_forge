import asyncio
import concurrent.futures
import copy
import datetime
import functools
import os
import re
import sys
import threading
import warnings
from base64 import b64decode, b64encode
from queue import Empty
from typing import Any
from unittest.mock import MagicMock, Mock
import nbformat
import pytest
import xmltodict
from flaky import flaky  # type:ignore
from jupyter_client import KernelClient, KernelManager
from jupyter_client._version import version_info
from jupyter_client.kernelspec import KernelSpecManager
from nbconvert.filters import strip_ansi
from nbformat import NotebookNode
from testpath import modified_env
from traitlets import TraitError
from nbclient import NotebookClient, execute
from nbclient.exceptions import CellExecutionError
from .base import NBClientTestsBase
def test_timeout_func(self):
    """Check that an error is raised when a computation times out"""
    filename = os.path.join(current_dir, 'files', 'Interrupt.ipynb')
    res = self.build_resources()
    res['metadata']['path'] = os.path.dirname(filename)

    def timeout_func(source):
        return 10
    with pytest.raises(TimeoutError):
        run_notebook(filename, {'timeout_func': timeout_func}, res)