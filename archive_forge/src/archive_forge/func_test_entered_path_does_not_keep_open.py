import sys
import unittest
import importlib_resources as resources
import pathlib
from . import data01
from . import util
from importlib import import_module
def test_entered_path_does_not_keep_open(self):
    """
        Mimic what certifi does on import to make its bundle
        available for the process duration.
        """
    resources.as_file(resources.files('data01') / 'binary.file').__enter__()