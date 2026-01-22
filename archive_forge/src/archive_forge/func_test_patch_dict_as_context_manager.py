import os
import sys
from collections import OrderedDict
import unittest
from unittest.test.testmock import support
from unittest.test.testmock.support import SomeClass, is_instance
from test.test_importlib.util import uncache
from unittest.mock import (
def test_patch_dict_as_context_manager(self):
    foo = {'a': 'b'}
    with patch.dict(foo, a='c') as patched:
        self.assertEqual(patched, {'a': 'c'})
    self.assertEqual(foo, {'a': 'b'})