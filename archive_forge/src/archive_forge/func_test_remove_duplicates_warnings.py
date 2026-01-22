from __future__ import annotations
import sys
from importlib.metadata import EntryPoint
from unittest import mock
import pytest
from xarray.backends import common, plugins
from xarray.tests import (
importlib_metadata_mock = "importlib.metadata"
def test_remove_duplicates_warnings(dummy_duplicated_entrypoints) -> None:
    with pytest.warns(RuntimeWarning) as record:
        _ = plugins.remove_duplicates(dummy_duplicated_entrypoints)
    assert len(record) == 2
    message0 = str(record[0].message)
    message1 = str(record[1].message)
    assert 'entrypoints' in message0
    assert 'entrypoints' in message1