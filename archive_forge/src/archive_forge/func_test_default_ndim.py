from Cython.TestUtils import CythonTest
import Cython.Compiler.Errors as Errors
from Cython.Compiler.Nodes import *
from Cython.Compiler.ParseTreeTransforms import *
from Cython.Compiler.Buffer import *
def test_default_ndim(self):
    self.parse(u'cdef int[:,:,:,:,:] x')
    self.parse(u'cdef unsigned long int[:,:,:,:,:] x')
    self.parse(u'cdef unsigned int[:,:,:,:,:] x')