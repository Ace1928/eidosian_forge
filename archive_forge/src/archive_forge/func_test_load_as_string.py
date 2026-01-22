import unittest
from Cython.Compiler import Code, UtilityCode
def test_load_as_string(self):
    got = strip_2tup(self.cls.load_as_string(self.name, self.filename, context=self.context))
    self.assertEqual(got, self.expected_tempita)