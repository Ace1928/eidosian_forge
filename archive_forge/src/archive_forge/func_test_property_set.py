import pytest
from string import ascii_letters
from random import randint
import gc
import sys
@pytest.mark.parametrize('name,args,kwargs,val,reset_val', [('NumericProperty', (0,), {}, 10, 0), ('NumericProperty', (0,), {}, '10dp', 0), ('NumericProperty', (0,), {}, [10, 'dp'], 0), ('ObjectProperty', (None,), {}, 5, 0), ('VariableListProperty', ([0, 0, 0, 0],), {}, [2, 4], [0]), ('BoundedNumericProperty', (1,), {'min': 0, 'max': 2}, 0.5, 1), ('DictProperty', ({},), {}, {'name': 1}, {}), ('ColorProperty', ([1, 1, 1, 1],), {}, 'red', [1, 1, 1, 1]), ('BooleanProperty', (False,), {}, True, False), ('OptionProperty', ('a',), {'options': ['a', 'b']}, 'b', 'a'), ('StringProperty', ('',), {}, 'a', ''), ('ListProperty', ([],), {}, [1, 2], []), ('AliasProperty', (0,), {}, 1, 0), ('ReferenceListProperty', ((1, 2),), {}, (3, 4), (1, 2))])
@pytest.mark.parametrize('exclude_first', [True, False])
def test_property_set(kivy_benchmark, name, args, kwargs, val, reset_val, exclude_first):
    event_cls = get_event_class(name, args, kwargs)
    e = event_cls()

    def set_property():
        e.a = reset_val
        e.a = val
    if exclude_first:
        set_property()
    kivy_benchmark(set_property)