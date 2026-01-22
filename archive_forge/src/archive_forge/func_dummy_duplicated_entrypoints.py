from __future__ import annotations
import sys
from importlib.metadata import EntryPoint
from unittest import mock
import pytest
from xarray.backends import common, plugins
from xarray.tests import (
importlib_metadata_mock = "importlib.metadata"
@pytest.fixture
def dummy_duplicated_entrypoints():
    specs = [['engine1', 'xarray.tests.test_plugins:backend_1', 'xarray.backends'], ['engine1', 'xarray.tests.test_plugins:backend_2', 'xarray.backends'], ['engine2', 'xarray.tests.test_plugins:backend_1', 'xarray.backends'], ['engine2', 'xarray.tests.test_plugins:backend_2', 'xarray.backends']]
    eps = [EntryPoint(name, value, group) for name, value, group in specs]
    return eps