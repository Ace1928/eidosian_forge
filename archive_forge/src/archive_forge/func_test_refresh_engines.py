from __future__ import annotations
import sys
from importlib.metadata import EntryPoint
from unittest import mock
import pytest
from xarray.backends import common, plugins
from xarray.tests import (
importlib_metadata_mock = "importlib.metadata"
def test_refresh_engines() -> None:
    from xarray.backends import list_engines, refresh_engines
    EntryPointMock1 = mock.MagicMock()
    EntryPointMock1.name = 'test1'
    EntryPointMock1.load.return_value = DummyBackendEntrypoint1
    if sys.version_info >= (3, 10):
        return_value = EntryPoints([EntryPointMock1])
    else:
        return_value = {'xarray.backends': [EntryPointMock1]}
    with mock.patch('xarray.backends.plugins.entry_points', return_value=return_value):
        list_engines.cache_clear()
        engines = list_engines()
    assert 'test1' in engines
    assert isinstance(engines['test1'], DummyBackendEntrypoint1)
    EntryPointMock2 = mock.MagicMock()
    EntryPointMock2.name = 'test2'
    EntryPointMock2.load.return_value = DummyBackendEntrypoint2
    if sys.version_info >= (3, 10):
        return_value2 = EntryPoints([EntryPointMock2])
    else:
        return_value2 = {'xarray.backends': [EntryPointMock2]}
    with mock.patch('xarray.backends.plugins.entry_points', return_value=return_value2):
        refresh_engines()
        engines = list_engines()
    assert 'test1' not in engines
    assert 'test2' in engines
    assert isinstance(engines['test2'], DummyBackendEntrypoint2)
    refresh_engines()