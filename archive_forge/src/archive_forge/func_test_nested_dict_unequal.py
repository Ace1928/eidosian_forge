from unittest import TestCase
from jsonschema._utils import equal
def test_nested_dict_unequal(self):
    dict_1 = {'a': {'a': 'b', 'c': 'd'}, 'c': 'd'}
    dict_2 = {'c': 'd', 'a': {'a': 'b', 'c': 'x'}}
    self.assertFalse(equal(dict_1, dict_2))