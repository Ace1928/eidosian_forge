import unittest
from unittest import mock
from traits.api import Bool, HasTraits, Int, Instance
from traits.observation._named_trait_observer import (
from traits.observation._observer_graph import ObserverGraph
from traits.observation._testing import (
def test_ordinary_has_traits(self):
    observer = create_observer(name='value1', optional=False)
    foo = ClassWithTwoValue()
    actual = list(observer.iter_observables(foo))
    self.assertEqual(actual, [foo._trait('value1', 2)])