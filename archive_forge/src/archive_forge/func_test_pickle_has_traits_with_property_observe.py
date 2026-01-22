import io
import pickle
import unittest
from unittest import mock
import weakref
from traits.api import (
from traits.trait_base import Undefined
from traits.observation.api import (
def test_pickle_has_traits_with_property_observe(self):
    instance = ClassWithPropertyMultipleObserves()
    for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
        serialized = pickle.dumps(instance, protocol=protocol)
        deserialized = pickle.loads(serialized)
        handler = mock.Mock()
        deserialized.observe(handler, 'computed_value')
        deserialized.age = 1
        self.assertEqual(handler.call_count, 1)