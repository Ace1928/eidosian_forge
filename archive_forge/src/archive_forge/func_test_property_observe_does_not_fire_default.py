import io
import pickle
import unittest
from unittest import mock
import weakref
from traits.api import (
from traits.trait_base import Undefined
from traits.observation.api import (
def test_property_observe_does_not_fire_default(self):
    instance_observe = ClassWithPropertyObservesDefault()
    handler_observe = mock.Mock()
    instance_observe.observe(handler_observe, 'extended_age')
    instance_depends_on = ClassWithPropertyDependsOnDefault()
    instance_depends_on.on_trait_change(get_otc_handler(mock.Mock()), 'extended_age')
    self.assertFalse(instance_observe.info_with_default_computed)
    self.assertTrue(instance_depends_on.info_with_default_computed)