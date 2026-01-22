import sys
import unittest
import importlib_resources as resources
import pathlib
from . import data01
from . import util
from importlib import import_module
def test_unrelated_contents(self):
    """
        Test thata zip with two unrelated subpackages return
        distinct resources. Ref python/importlib_resources#44.
        """
    self.assertEqual(names(resources.files('data02.one')), {'__init__.py', 'resource1.txt'})
    self.assertEqual(names(resources.files('data02.two')), {'__init__.py', 'resource2.txt'})