from Cython.TestUtils import CythonTest
import Cython.Compiler.Errors as Errors
from Cython.Compiler.Nodes import *
from Cython.Compiler.ParseTreeTransforms import *
from Cython.Compiler.Buffer import *
def disable_test_parse_sizeof(self):
    self.parse(u'sizeof(int[NN])')
    self.parse(u'sizeof(int[])')
    self.parse(u'sizeof(int[][NN])')
    self.not_parseable(u'Expected an identifier or literal', u'sizeof(int[:NN])')
    self.not_parseable(u"Expected ']'", u'sizeof(foo[dtype=bar]')