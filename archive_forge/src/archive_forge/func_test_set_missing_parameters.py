from __future__ import annotations
import sys
from importlib.metadata import EntryPoint
from unittest import mock
import pytest
from xarray.backends import common, plugins
from xarray.tests import (
importlib_metadata_mock = "importlib.metadata"
def test_set_missing_parameters() -> None:
    backend_1 = DummyBackendEntrypoint1
    backend_2 = DummyBackendEntrypoint2
    backend_2.open_dataset_parameters = ('filename_or_obj',)
    engines = {'engine_1': backend_1, 'engine_2': backend_2}
    plugins.set_missing_parameters(engines)
    assert len(engines) == 2
    assert backend_1.open_dataset_parameters == ('filename_or_obj', 'decoder')
    assert backend_2.open_dataset_parameters == ('filename_or_obj',)
    backend_kwargs = DummyBackendEntrypointKwargs
    backend_kwargs.open_dataset_parameters = ('filename_or_obj', 'decoder')
    plugins.set_missing_parameters({'engine': backend_kwargs})
    assert backend_kwargs.open_dataset_parameters == ('filename_or_obj', 'decoder')
    backend_args = DummyBackendEntrypointArgs
    backend_args.open_dataset_parameters = ('filename_or_obj', 'decoder')
    plugins.set_missing_parameters({'engine': backend_args})
    assert backend_args.open_dataset_parameters == ('filename_or_obj', 'decoder')
    backend_1.open_dataset_parameters = None
    backend_1.open_dataset_parameters = None
    backend_kwargs.open_dataset_parameters = None
    backend_args.open_dataset_parameters = None