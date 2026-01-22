import io
import pickle
import unittest
from unittest import mock
import weakref
from traits.api import (
from traits.trait_base import Undefined
from traits.observation.api import (
def test_property_with_cache(self):
    instance = ClassWithPropertyObservesWithCache()
    handler = mock.Mock()
    instance.observe(handler, 'discounted')
    instance.age = 1
    (event,), _ = handler.call_args_list[-1]
    self.assertIs(event.object, instance)
    self.assertEqual(event.name, 'discounted')
    self.assertIs(event.old, Undefined)
    self.assertIs(event.new, False)
    handler.reset_mock()
    instance.age = 80
    (event,), _ = handler.call_args_list[-1]
    self.assertIs(event.object, instance)
    self.assertEqual(event.name, 'discounted')
    self.assertIs(event.old, False)
    self.assertIs(event.new, True)