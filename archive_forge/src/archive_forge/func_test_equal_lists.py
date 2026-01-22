from unittest import TestCase
from jsonschema._utils import equal
def test_equal_lists(self):
    list_1 = ['a', 'b', 'c']
    list_2 = ['a', 'b', 'c']
    self.assertTrue(equal(list_1, list_2))