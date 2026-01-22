from unittest import TestCase
from jsonschema._utils import equal
def test_mixed_nested_equal(self):
    dict_1 = {'a': ['a', 'b', 'c', 'd'], 'c': 'd'}
    dict_2 = {'c': 'd', 'a': ['a', 'b', 'c', 'd']}
    self.assertTrue(equal(dict_1, dict_2))