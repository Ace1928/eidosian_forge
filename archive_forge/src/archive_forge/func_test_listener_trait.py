import copy
import pickle
import unittest
from traits.has_traits import (
from traits.ctrait import CTrait
from traits.observation.api import (
from traits.observation.exception_handling import (
from traits.traits import ForwardProperty, generic_trait
from traits.trait_types import Event, Float, Instance, Int, List, Map, Str
from traits.trait_errors import TraitError
def test_listener_trait(self):

    @on_trait_change('something')
    def listener(self):
        pass
    class_name = 'MyClass'
    bases = (object,)
    class_dict = {'attr': 'something', 'my_listener': listener}
    update_traits_class_dict(class_name, bases, class_dict)
    self.assertEqual(class_dict[BaseTraits], {})
    self.assertEqual(class_dict[ClassTraits], {})
    self.assertEqual(class_dict[InstanceTraits], {})
    self.assertEqual(class_dict[ListenerTraits], {'my_listener': ('method', {'pattern': 'something', 'post_init': False, 'dispatch': 'same'})})