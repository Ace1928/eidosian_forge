import unittest
from unittest import mock
from traits.has_traits import HasTraits
from traits.observation._set_item_observer import SetItemObserver
from traits.observation._testing import (
from traits.trait_set_object import TraitSet
from traits.trait_types import Set
def test_not_a_set(self):
    observer = create_observer(optional=False)
    with self.assertRaises(ValueError) as exception_context:
        list(observer.iter_observables(None))
    self.assertIn('Expected a TraitSet to be observed, got', str(exception_context.exception))