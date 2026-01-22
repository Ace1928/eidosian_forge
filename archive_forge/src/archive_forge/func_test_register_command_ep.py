from __future__ import annotations
import importlib
import json
import pathlib
import platform
import sys
import click
import pytest
import yaml
from click.testing import CliRunner
import dask
import dask.cli
from dask._compatibility import importlib_metadata
def test_register_command_ep():
    from dask.cli import _register_command_ep
    bad_ep = importlib_metadata.EntryPoint(name='bad', value='dask.tests.test_cli:bad_command', group='dask_cli')
    good_ep = importlib_metadata.EntryPoint(name='good', value='dask.tests.test_cli:good_command', group='dask_cli')

    class ErrorEP:

        @property
        def name(self):
            return 'foo'

        def load(self):
            raise ImportError('Entrypoint could not be imported')
    with pytest.warns(UserWarning, match='must be instances of'):
        _register_command_ep(dummy_cli, bad_ep)
    with pytest.warns(UserWarning, match='exception occurred'):
        _register_command_ep(dummy_cli, ErrorEP())
    _register_command_ep(dummy_cli, good_ep)
    assert 'good' in dummy_cli.commands
    assert dummy_cli.commands['good'] is good_command