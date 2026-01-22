import os
import tempfile
import unittest
from Cython.Shadow import inline
from Cython.Build.Inline import safe_type
from Cython.TestUtils import CythonTest
def test_class_ref(self):

    class Type(object):
        pass
    tp = inline('Type')['Type']
    self.assertEqual(tp, Type)