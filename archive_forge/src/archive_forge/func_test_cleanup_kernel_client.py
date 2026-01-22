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
def test_cleanup_kernel_client(self):
    filename = os.path.join(current_dir, 'files', 'HelloWorld.ipynb')
    with open(filename) as f:
        input_nb = nbformat.read(f, 4)
    executor = NotebookClient(input_nb, resources=self.build_resources())
    executor.execute()
    assert executor.kc is None
    executor.execute(cleanup_kc=False)
    assert executor.kc is not None