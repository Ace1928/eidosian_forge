from unittest import TestCase
from jsonschema._utils import equal
def test_first_list_larger(self):
    list_1 = ['a', 'b', 'c']
    list_2 = ['a', 'b']
    self.assertFalse(equal(list_1, list_2))