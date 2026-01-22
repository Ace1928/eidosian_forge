from sympy.parsing.sym_expr import SymPyExpression
from sympy.testing.pytest import raises, XFAIL
from sympy.external import import_module
def test_paren_expr():
    c_src1 = 'int a = (1);int b = (1 + 2 * 3);'
    c_src2 = 'int a = 1, b = 2, c = 3;int d = (a);int e = (a + 1);int f = (a + b * c - d / e);'
    res1 = SymPyExpression(c_src1, 'c').return_expr()
    res2 = SymPyExpression(c_src2, 'c').return_expr()
    assert res1[0] == Declaration(Variable(Symbol('a'), type=IntBaseType(String('intc')), value=Integer(1)))
    assert res1[1] == Declaration(Variable(Symbol('b'), type=IntBaseType(String('intc')), value=Integer(7)))
    assert res2[0] == Declaration(Variable(Symbol('a'), type=IntBaseType(String('intc')), value=Integer(1)))
    assert res2[1] == Declaration(Variable(Symbol('b'), type=IntBaseType(String('intc')), value=Integer(2)))
    assert res2[2] == Declaration(Variable(Symbol('c'), type=IntBaseType(String('intc')), value=Integer(3)))
    assert res2[3] == Declaration(Variable(Symbol('d'), type=IntBaseType(String('intc')), value=Symbol('a')))
    assert res2[4] == Declaration(Variable(Symbol('e'), type=IntBaseType(String('intc')), value=Add(Symbol('a'), Integer(1))))
    assert res2[5] == Declaration(Variable(Symbol('f'), type=IntBaseType(String('intc')), value=Add(Symbol('a'), Mul(Symbol('b'), Symbol('c')), Mul(Integer(-1), Symbol('d'), Pow(Symbol('e'), Integer(-1))))))