import os
import sys
from collections import OrderedDict
import unittest
from unittest.test.testmock import support
from unittest.test.testmock.support import SomeClass, is_instance
from test.test_importlib.util import uncache
from unittest.mock import (
def test_cant_set_kwargs_when_passing_a_mock(self):

    @patch('unittest.test.testmock.support.X', new=object(), x=1)
    def test():
        pass
    with self.assertRaises(TypeError):
        test()