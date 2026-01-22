import os
import sys
from collections import OrderedDict
import unittest
from unittest.test.testmock import support
from unittest.test.testmock.support import SomeClass, is_instance
from test.test_importlib.util import uncache
from unittest.mock import (
def test_patchobject_with_string_as_target(self):
    msg = "'Something' must be the actual object to be patched, not a str"
    with self.assertRaisesRegex(TypeError, msg):
        patch.object('Something', 'do_something')