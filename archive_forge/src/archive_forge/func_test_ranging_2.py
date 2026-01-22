import unittest
from idna.intranges import intranges_from_list, intranges_contain, _encode_range
def test_ranging_2(self):
    self.assertEqual(intranges_from_list([111]), (_encode_range(111, 112),))