from unittest import TestCase
from jsonschema._utils import equal
def test_additional_key(self):
    dict_1 = {'a': 'b', 'c': 'd'}
    dict_2 = {'c': 'd', 'a': 'b', 'x': 'x'}
    self.assertFalse(equal(dict_1, dict_2))