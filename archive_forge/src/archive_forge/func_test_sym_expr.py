from sympy.testing.pytest import raises
from sympy.parsing.sym_expr import SymPyExpression
from sympy.external import import_module
def test_sym_expr():
    src1 = src + '            d = a + b -c\n            '
    expr3 = SymPyExpression(src, 'f')
    expr4 = SymPyExpression(src1, 'f')
    ls1 = expr3.return_expr()
    ls2 = expr4.return_expr()
    for i in range(0, 7):
        assert isinstance(ls1[i], Declaration)
        assert isinstance(ls2[i], Declaration)
    assert isinstance(ls2[8], Assignment)
    assert ls1[0] == Declaration(Variable(Symbol('a'), type=IntBaseType(String('integer')), value=Integer(0)))
    assert ls1[1] == Declaration(Variable(Symbol('b'), type=IntBaseType(String('integer')), value=Integer(0)))
    assert ls1[2] == Declaration(Variable(Symbol('c'), type=IntBaseType(String('integer')), value=Integer(0)))
    assert ls1[3] == Declaration(Variable(Symbol('d'), type=IntBaseType(String('integer')), value=Integer(0)))
    assert ls1[4] == Declaration(Variable(Symbol('p'), type=FloatBaseType(String('real')), value=Float(0.0)))
    assert ls1[5] == Declaration(Variable(Symbol('q'), type=FloatBaseType(String('real')), value=Float(0.0)))
    assert ls1[6] == Declaration(Variable(Symbol('r'), type=FloatBaseType(String('real')), value=Float(0.0)))
    assert ls1[7] == Declaration(Variable(Symbol('s'), type=FloatBaseType(String('real')), value=Float(0.0)))
    assert ls2[8] == Assignment(Variable(Symbol('d')), Symbol('a') + Symbol('b') - Symbol('c'))