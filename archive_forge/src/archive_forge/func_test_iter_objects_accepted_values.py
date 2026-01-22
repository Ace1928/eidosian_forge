import unittest
from unittest import mock
from traits.api import (
from traits.observation import _has_traits_helpers as helpers
from traits.observation import expression
from traits.observation.observe import observe
def test_iter_objects_accepted_values(self):
    foo = Foo(instance=Bar(), list_of_int=[1, 2])
    actual = list(helpers.iter_objects(foo, 'instance'))
    self.assertEqual(actual, [foo.instance])