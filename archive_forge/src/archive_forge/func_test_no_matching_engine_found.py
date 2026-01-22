from __future__ import annotations
import sys
from importlib.metadata import EntryPoint
from unittest import mock
import pytest
from xarray.backends import common, plugins
from xarray.tests import (
importlib_metadata_mock = "importlib.metadata"
@mock.patch('xarray.backends.plugins.list_engines', mock.MagicMock(return_value={'dummy': DummyBackendEntrypointArgs()}))
def test_no_matching_engine_found() -> None:
    with pytest.raises(ValueError, match='did not find a match in any'):
        plugins.guess_engine('not-valid')
    with pytest.raises(ValueError, match='found the following matches with the input'):
        plugins.guess_engine('foo.nc')