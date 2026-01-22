import io
import pickle
import unittest
from unittest import mock
import weakref
from traits.api import (
from traits.trait_base import Undefined
from traits.observation.api import (
def test_property_default_initializer_with_value_in_init(self):
    with self.assertRaises(AttributeError):
        ClassWithPropertyDependsOnInit(info_without_default=PersonInfo(age=30))
    instance = ClassWithPropertyObservesInit(info_without_default=PersonInfo(age=30))
    handler = mock.Mock()
    instance.observe(handler, 'extended_age')
    self.assertFalse(instance.sample_info_default_computed)
    self.assertEqual(instance.sample_info.age, 30)
    self.assertEqual(instance.extended_age, 30)
    self.assertEqual(handler.call_count, 0)
    instance.sample_info.age = 40
    self.assertEqual(handler.call_count, 1)
    instance_no_property = ClassWithInstanceDefaultInit(info_without_default=PersonInfo(age=30))
    self.assertFalse(instance_no_property.sample_info_default_computed)
    self.assertEqual(instance_no_property.sample_info.age, 30)