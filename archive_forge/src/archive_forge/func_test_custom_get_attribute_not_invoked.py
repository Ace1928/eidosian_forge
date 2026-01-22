import inspect
import keyword
import unittest
from collections import namedtuple
from unittest import mock
from bpython import autocomplete, inspection
from bpython.line import LinePart
def test_custom_get_attribute_not_invoked(self):
    com = autocomplete.AttrCompletion()
    self.assertSetEqual(com.matches(2, 'a.', locals_={'a': OverriddenGetattribute()}), {'a.b', 'a.a', 'a.method'})