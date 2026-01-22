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
@prepare_cell_mocks()
def test_cell_hooks(self, executor, cell_mock, message_mock):
    executor, hooks = get_executor_with_hooks(executor=executor)
    executor.execute_cell(cell_mock, 0)
    hooks['on_cell_start'].assert_called_once_with(cell=cell_mock, cell_index=0)
    hooks['on_cell_execute'].assert_called_once_with(cell=cell_mock, cell_index=0)
    hooks['on_cell_complete'].assert_called_once_with(cell=cell_mock, cell_index=0)
    hooks['on_cell_executed'].assert_called_once_with(cell=cell_mock, cell_index=0, execute_reply=EXECUTE_REPLY_OK)
    hooks['on_cell_error'].assert_not_called()
    hooks['on_notebook_start'].assert_not_called()
    hooks['on_notebook_complete'].assert_not_called()
    hooks['on_notebook_error'].assert_not_called()