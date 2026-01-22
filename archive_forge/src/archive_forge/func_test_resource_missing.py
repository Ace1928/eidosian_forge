import sys
import unittest
import importlib_resources as resources
import pathlib
from . import data01
from . import util
from importlib import import_module
def test_resource_missing(self):
    package = util.create_package(file=data01, path=data01.__file__, contents=['A', 'B', 'C', 'D/E', 'D/F'])
    self.assertFalse(resources.files(package).joinpath('Z').is_file())