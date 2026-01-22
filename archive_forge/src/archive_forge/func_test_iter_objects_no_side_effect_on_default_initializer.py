import unittest
from unittest import mock
from traits.api import Bool, HasTraits, Int, Instance
from traits.observation._named_trait_observer import (
from traits.observation._observer_graph import ObserverGraph
from traits.observation._testing import (
def test_iter_objects_no_side_effect_on_default_initializer(self):
    observer = create_observer(name='instance')
    foo = ClassWithDefault()
    actual = list(observer.iter_objects(foo))
    self.assertEqual(actual, [])
    self.assertNotIn('instance', foo.__dict__)
    self.assertFalse(foo.instance_default_calculated, 'Unexpected side-effect on the default initializer.')