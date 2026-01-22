from Cython.TestUtils import CythonTest
from Cython.Compiler.TreeFragment import *
from Cython.Compiler.Nodes import *
from Cython.Compiler.UtilNodes import *
def test_substitution(self):
    F = self.fragment(u'x = 4')
    y = NameNode(pos=None, name=u'y')
    T = F.substitute({'x': y})
    self.assertCode(u'y = 4', T)