import copy
import re
import sys
import tempfile
from test.support import ALWAYS_EQ
import unittest
from unittest.test.testmock.support import is_instance
from unittest import mock
from unittest.mock import (
def test_bool_not_called_when_passing_spec_arg(self):

    class Something:

        def __init__(self):
            self.obj_with_bool_func = unittest.mock.MagicMock()
    obj = Something()
    with unittest.mock.patch.object(obj, 'obj_with_bool_func', spec=object):
        pass
    self.assertEqual(obj.obj_with_bool_func.__bool__.call_count, 0)