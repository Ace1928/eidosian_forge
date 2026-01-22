import unittest
from unittest import mock
from traits.has_traits import HasTraits
from traits.trait_base import Undefined, Uninitialized
from traits.trait_types import Float, Instance, Int, List
from traits.observation._filtered_trait_observer import FilteredTraitObserver
from traits.observation._testing import (
def test_not_equal_filter(self):
    observer1 = FilteredTraitObserver(notify=True, filter=DummyFilter(return_value=True))
    observer2 = FilteredTraitObserver(notify=True, filter=DummyFilter(return_value=False))
    self.assertNotEqual(observer1, observer2)