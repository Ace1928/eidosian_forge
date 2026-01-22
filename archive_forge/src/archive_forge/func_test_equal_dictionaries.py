from unittest import TestCase
from jsonschema._utils import equal
def test_equal_dictionaries(self):
    dict_1 = {'a': 'b', 'c': 'd'}
    dict_2 = {'c': 'd', 'a': 'b'}
    self.assertTrue(equal(dict_1, dict_2))