import copy
import re
import sys
import tempfile
from test.support import ALWAYS_EQ
import unittest
from unittest.test.testmock.support import is_instance
from unittest import mock
from unittest.mock import (
def test_assert_has_calls_nested_without_spec(self):
    m = MagicMock()
    m().foo().bar().baz()
    m.one().two().three()
    calls = call.one().two().three().call_list()
    m.assert_has_calls(calls)