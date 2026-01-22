import os
import sys
from collections import OrderedDict
import unittest
from unittest.test.testmock import support
from unittest.test.testmock.support import SomeClass, is_instance
from test.test_importlib.util import uncache
from unittest.mock import (
def test_invalid_target(self):

    class Foo:
        pass
    for target in ['', 12, Foo()]:
        with self.subTest(target=target):
            with self.assertRaises(TypeError):
                patch(target)