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
def assert_notebooks_equal(expected, actual):
    expected_cells = expected['cells']
    actual_cells = actual['cells']
    assert len(expected_cells) == len(actual_cells)
    for expected_cell, actual_cell in zip(expected_cells, actual_cells):
        expected_outputs = expected_cell.get('outputs', [])
        actual_outputs = actual_cell.get('outputs', [])
        normalized_expected_outputs = list(map(normalize_output, expected_outputs))
        normalized_actual_outputs = list(map(normalize_output, actual_outputs))
        assert normalized_expected_outputs == normalized_actual_outputs
        expected_execution_count = expected_cell.get('execution_count', None)
        actual_execution_count = actual_cell.get('execution_count', None)
        assert expected_execution_count == actual_execution_count