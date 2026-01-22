from Cython.TestUtils import CythonTest
import Cython.Compiler.Errors as Errors
from Cython.Compiler.Nodes import *
from Cython.Compiler.ParseTreeTransforms import *
from Cython.Compiler.Buffer import *
def test_non_slice_memview(self):
    self.not_parseable(u"An axis specification in memoryview declaration does not have a ':'.", u'cdef double[:foo, bar] x')
    self.not_parseable(u"An axis specification in memoryview declaration does not have a ':'.", u'cdef double[0:foo, bar] x')