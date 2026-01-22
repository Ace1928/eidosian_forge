from unittest import TestCase
from jsonschema._utils import equal
def test_same_list(self):
    list_1 = ['a', 'b', 'c']
    self.assertTrue(equal(list_1, list_1))