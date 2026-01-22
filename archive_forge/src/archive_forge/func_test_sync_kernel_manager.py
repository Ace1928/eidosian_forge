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
def test_sync_kernel_manager(self):
    nb = nbformat.v4.new_notebook()
    executor = NotebookClient(nb, kernel_name='python', kernel_manager_class=KernelManager)
    nb = executor.execute()
    assert 'language_info' in nb.metadata
    with executor.setup_kernel():
        assert executor.kc is not None
        info_msg = executor.wait_for_reply(executor.kc.kernel_info())
        assert info_msg is not None
        assert 'name' in info_msg['content']['language_info']