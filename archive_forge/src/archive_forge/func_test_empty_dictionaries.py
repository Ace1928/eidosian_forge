from unittest import TestCase
from jsonschema._utils import equal
def test_empty_dictionaries(self):
    dict_1 = {}
    dict_2 = {}
    self.assertTrue(equal(dict_1, dict_2))