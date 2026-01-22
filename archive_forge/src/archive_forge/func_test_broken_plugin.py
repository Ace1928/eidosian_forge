from __future__ import annotations
import sys
from importlib.metadata import EntryPoint
from unittest import mock
import pytest
from xarray.backends import common, plugins
from xarray.tests import (
importlib_metadata_mock = "importlib.metadata"
def test_broken_plugin() -> None:
    broken_backend = EntryPoint('broken_backend', 'xarray.tests.test_plugins:backend_1', 'xarray.backends')
    with pytest.warns(RuntimeWarning) as record:
        _ = plugins.build_engines(EntryPoints([broken_backend]))
    assert len(record) == 1
    message = str(record[0].message)
    assert "Engine 'broken_backend'" in message