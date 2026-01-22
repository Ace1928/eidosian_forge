import unittest
from unittest import mock
from traits.api import (
from traits.observation import _has_traits_helpers as helpers
from traits.observation import expression
from traits.observation.observe import observe
def test_property_equality_no_effect(self):
    instance = ObjectWithEqualityComparisonMode()
    instance.number = 1
    handler = mock.Mock()
    observe(object=instance, expression=expression.trait('calculated'), handler=handler)
    instance.number = 2
    self.assertEqual(handler.call_count, 1)