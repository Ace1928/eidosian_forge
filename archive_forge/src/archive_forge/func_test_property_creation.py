import pytest
from string import ascii_letters
from random import randint
import gc
import sys
@pytest.mark.parametrize('name,args,kwargs', [('NumericProperty', (0,), {}), ('ObjectProperty', (None,), {}), ('VariableListProperty', ([0, 0, 0, 0],), {}), ('BoundedNumericProperty', (1,), {'min': 0, 'max': 2}), ('DictProperty', ({},), {}), ('ColorProperty', ([1, 1, 1, 1],), {}), ('BooleanProperty', (False,), {}), ('OptionProperty', ('a',), {'options': ['a', 'b']}), ('StringProperty', ('',), {}), ('ListProperty', ([],), {}), ('AliasProperty', (), {}), ('ReferenceListProperty', (), {})])
def test_property_creation(kivy_benchmark, name, args, kwargs):
    event_cls = get_event_class(name, args, kwargs)
    e = event_cls()
    kivy_benchmark(event_cls)