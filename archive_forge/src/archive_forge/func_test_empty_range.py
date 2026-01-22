import unittest
from idna.intranges import intranges_from_list, intranges_contain, _encode_range
def test_empty_range(self):
    self.assertEqual(intranges_from_list([]), ())