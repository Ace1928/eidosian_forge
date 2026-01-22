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
def test_process_message_wrapper(self):
    outputs: list = []

    class WrappedPreProc(NotebookClient):

        def process_message(self, msg, cell, cell_index):
            result = super().process_message(msg, cell, cell_index)
            if result:
                outputs.append(result)
            return result
    current_dir = os.path.dirname(__file__)
    filename = os.path.join(current_dir, 'files', 'HelloWorld.ipynb')
    with open(filename) as f:
        input_nb = nbformat.read(f, 4)
    original = copy.deepcopy(input_nb)
    wpp = WrappedPreProc(input_nb)
    executed = wpp.execute()
    assert outputs == [{'name': 'stdout', 'output_type': 'stream', 'text': 'Hello World\n'}]
    assert_notebooks_equal(original, executed)