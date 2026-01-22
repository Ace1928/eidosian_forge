import sys
import unittest
import importlib_resources as resources
import pathlib
from . import data01
from . import util
from importlib import import_module
def test_as_file_directory(self):
    with resources.as_file(resources.files('data01')) as data:
        assert data.name == 'data01'
        assert data.is_dir()
        assert data.joinpath('subdirectory').is_dir()
        assert len(list(data.iterdir()))
    assert not data.parent.exists()