import unittest
from Cython.Compiler import PyrexTypes as pt
from Cython.Compiler.ExprNodes import NameNode
from Cython.Compiler.PyrexTypes import CFuncTypeArg
def test_cpp_reference_two_args(self):
    function_types = [cfunctype(pt.CReferenceType(pt.c_int_type), pt.CReferenceType(pt.c_long_type)), cfunctype(pt.CReferenceType(pt.c_long_type), pt.CReferenceType(pt.c_long_type))]
    functions = [NameNode(None, type=t) for t in function_types]
    self.assertMatches(function_types[0], [pt.c_int_type, pt.c_long_type], functions)
    self.assertMatches(function_types[1], [pt.c_long_type, pt.c_long_type], functions)
    self.assertMatches(function_types[1], [pt.c_long_type, pt.c_int_type], functions)