import unittest
from unittest import mock
from traits.has_traits import HasTraits
from traits.observation._testing import (
from traits.observation._trait_added_observer import (
from traits.trait_types import Str
def test_notifier_inherited(self):
    notifier = DummyNotifier()
    wrapped_observer = DummyObserver(notifier=notifier)
    observer = _RestrictedNamedTraitObserver(name='name', wrapped_observer=wrapped_observer)
    self.assertEqual(observer.get_notifier(None, None, None), notifier)