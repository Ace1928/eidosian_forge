import unittest
from unittest import mock
from traits.api import Bool, HasTraits, Int, Instance
from traits.observation._named_trait_observer import (
from traits.observation._observer_graph import ObserverGraph
from traits.observation._testing import (
def test_iter_objects(self):
    observer = create_observer(name='instance')
    foo = ClassWithInstance(instance=ClassWithTwoValue())
    actual = list(observer.iter_objects(foo))
    self.assertEqual(actual, [foo.instance])