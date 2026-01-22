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
def test_observe_trait(self):

    @observe(trait('value'), post_init=True, dispatch='ui')
    @observe('name')
    def handler(self, event):
        pass
    class_name = 'MyClass'
    bases = (object,)
    class_dict = {'attr': 'something', 'my_listener': handler}
    update_traits_class_dict(class_name, bases, class_dict)
    self.assertEqual(class_dict[ObserverTraits], {'my_listener': [{'graphs': compile_str('name'), 'post_init': False, 'dispatch': 'same', 'handler_getter': getattr}, {'graphs': compile_expr(trait('value')), 'post_init': True, 'dispatch': 'ui', 'handler_getter': getattr}]})