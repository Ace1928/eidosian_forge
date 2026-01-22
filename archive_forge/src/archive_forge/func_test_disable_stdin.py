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
def test_disable_stdin(self):
    """Test disabling standard input"""
    filename = os.path.join(current_dir, 'files', 'Disable Stdin.ipynb')
    res = self.build_resources()
    res['metadata']['path'] = os.path.dirname(filename)
    input_nb, output_nb = run_notebook(filename, {'allow_errors': True}, res)
    self.assertEqual(len(output_nb['cells']), 1)
    self.assertEqual(len(output_nb['cells'][0]['outputs']), 1)
    output = output_nb['cells'][0]['outputs'][0]
    self.assertEqual(output['output_type'], 'error')
    self.assertEqual(output['ename'], 'StdinNotImplementedError')
    self.assertEqual(output['evalue'], 'raw_input was called, but this frontend does not support input requests.')