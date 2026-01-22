from Cython.TestUtils import CythonTest
import Cython.Compiler.Errors as Errors
from Cython.Compiler.Nodes import *
from Cython.Compiler.ParseTreeTransforms import *
from Cython.Compiler.Buffer import *
def test_zero_offset(self):
    self.parse(u'cdef long double[0:] x')
    self.parse(u'cdef int[0:] x')