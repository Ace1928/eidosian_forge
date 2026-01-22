import os
import sys
from collections import OrderedDict
import unittest
from unittest.test.testmock import support
from unittest.test.testmock.support import SomeClass, is_instance
from test.test_importlib.util import uncache
from unittest.mock import (
def test_patch_dict_decorator_resolution(self):
    original = support.target.copy()

    @patch.dict('unittest.test.testmock.support.target', {'bar': 'BAR'})
    def test():
        self.assertEqual(support.target, {'foo': 'BAZ', 'bar': 'BAR'})
    try:
        support.target = {'foo': 'BAZ'}
        test()
        self.assertEqual(support.target, {'foo': 'BAZ'})
    finally:
        support.target = original