from sympy.parsing.sym_expr import SymPyExpression
from sympy.testing.pytest import raises
from sympy.external import import_module
def test_c_parse():
    src1 = '        int a, b = 4;\n        float c, d = 2.4;\n        '
    expr1.convert_to_expr(src1, 'c')
    ls = expr1.return_expr()
    assert ls[0] == Declaration(Variable(Symbol('a'), type=IntBaseType(String('intc'))))
    assert ls[1] == Declaration(Variable(Symbol('b'), type=IntBaseType(String('intc')), value=Integer(4)))
    assert ls[2] == Declaration(Variable(Symbol('c'), type=FloatType(String('float32'), nbits=Integer(32), nmant=Integer(23), nexp=Integer(8))))
    assert ls[3] == Declaration(Variable(Symbol('d'), type=FloatType(String('float32'), nbits=Integer(32), nmant=Integer(23), nexp=Integer(8)), value=Float('2.3999999999999999', precision=53)))