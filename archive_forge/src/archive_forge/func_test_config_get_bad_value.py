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
def test_config_get_bad_value():
    runner = CliRunner()
    result = runner.invoke(dask.cli.config_get, ['bad_key'])
    assert result.exit_code != 0
    assert result.output.startswith('Section not found')