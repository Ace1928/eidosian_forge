import unittest
from unittest import mock
from traits.has_traits import HasTraits
from traits.observation._testing import (
from traits.observation._trait_added_observer import (
from traits.trait_types import Str
def test_not_equal_observer(self):
    observer1 = _RestrictedNamedTraitObserver(name='name', wrapped_observer=DummyObserver())
    observer2 = _RestrictedNamedTraitObserver(name='name', wrapped_observer=DummyObserver())
    self.assertNotEqual(observer1, observer2)