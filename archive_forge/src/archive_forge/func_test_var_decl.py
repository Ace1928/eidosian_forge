from sympy.parsing.sym_expr import SymPyExpression
from sympy.testing.pytest import raises, XFAIL
from sympy.external import import_module
@XFAIL
def test_var_decl():
    c_src1 = 'int b = 100;' + '\n' + 'int a = b;' + '\n'
    c_src2 = 'int a = 1;' + '\n' + 'int b = a + 1;' + '\n'
    c_src3 = 'float a = 10.0 + 2.5;' + '\n' + 'float b = a * 20.0;' + '\n'
    c_src4 = 'int a = 1 + 100 - 3 * 6;' + '\n'
    c_src5 = 'int a = (((1 + 100) * 12) - 3) * (6 - 10);' + '\n'
    c_src6 = 'int b = 2;' + '\n' + 'int c = 3;' + '\n' + 'int a = b + c * 4;' + '\n'
    c_src7 = 'int b = 1;' + '\n' + 'int c = b + 2;' + '\n' + 'int a = 10 * b * b * c;' + '\n'
    c_src8 = 'void func()' + '{' + '\n' + 'int a = 1;' + '\n' + 'int b = 2;' + '\n' + 'int temp = a;' + '\n' + 'a = b;' + '\n' + 'b = temp;' + '\n' + '}'
    c_src9 = 'int a = 1;' + '\n' + 'int b = 2;' + '\n' + 'int c = a;' + '\n' + 'int d = a + b + c;' + '\n' + 'int e = a*a*a + 3*a*a*b + 3*a*b*b + b*b*b;' + '\nint f = (a + b + c) * (a + b - c);' + '\n' + 'int g = (a + b + c + d)*(a + b + c + d)*(a * (b - c));' + '\n'
    c_src10 = 'float a = 10.0;' + '\n' + 'float b = 2.5;' + '\n' + 'float c = a*a + 2*a*b + b*b;' + '\n'
    c_src11 = 'float a = 10.0 / 2.5;' + '\n'
    c_src12 = 'int a = 100 / 4;' + '\n'
    c_src13 = 'int a = 20 - 100 / 4 * 5 + 10;' + '\n'
    c_src14 = 'int a = (20 - 100) / 4 * (5 + 10);' + '\n'
    c_src15 = 'int a = 4;' + '\n' + 'int b = 2;' + '\n' + 'float c = b/a;' + '\n'
    c_src16 = 'int a = 2;' + '\n' + 'int d = 5;' + '\n' + 'int n = 10;' + '\n' + 'int s = (a/2)*(2*a + (n-1)*d);' + '\n'
    c_src17 = 'int a = 1 % 2;' + '\n'
    c_src18 = 'int a = 2;' + '\n' + 'int b = a % 3;' + '\n'
    c_src19 = 'int a = 100;' + '\n' + 'int b = 3;' + '\n' + 'int c = a % b;' + '\n'
    c_src20 = 'int a = 100;' + '\n' + 'int b = 3;' + '\n' + 'int mod = 1000000007;' + '\n' + 'int c = (a + b * (100/a)) % mod;' + '\n'
    c_src21 = 'int a = 100;' + '\n' + 'int b = 3;' + '\n' + 'int mod = 1000000007;' + '\n' + 'int c = ((a % mod + b % mod) % mod *(a % mod - b % mod) % mod) % mod;' + '\n'
    c_src22 = 'bool a = 1 == 2, b = 1 != 2;'
    c_src23 = 'bool a = 1 < 2, b = 1 <= 2, c = 1 > 2, d = 1 >= 2;'
    c_src24 = 'int a = 1, b = 2;' + '\n' + 'bool c1 = a == 1;' + '\n' + 'bool c2 = b == 2;' + '\n' + 'bool c3 = 1 != a;' + '\n' + 'bool c4 = 1 != b;' + '\n' + 'bool c5 = a < 0;' + '\n' + 'bool c6 = b <= 10;' + '\n' + 'bool c7 = a > 0;' + '\n' + 'bool c8 = b >= 11;'
    c_src25 = 'int a = 3, b = 4;' + '\n' + 'bool c1 = a == b;' + '\n' + 'bool c2 = a != b;' + '\n' + 'bool c3 = a < b;' + '\n' + 'bool c4 = a <= b;' + '\n' + 'bool c5 = a > b;' + '\n' + 'bool c6 = a >= b;'
    c_src26 = 'float a = 1.25, b = 2.5;' + '\n' + 'bool c1 = a == 1.25;' + '\n' + 'bool c2 = b == 2.54;' + '\n' + 'bool c3 = 1.2 != a;' + '\n' + 'bool c4 = 1.5 != b;'
    c_src27 = 'float a = 1.25, b = 2.5;' + '\n' + 'bool c1 = a == b;' + '\n' + 'bool c2 = a != b;' + '\n' + 'bool c3 = a < b;' + '\n' + 'bool c4 = a <= b;' + '\n' + 'bool c5 = a > b;' + '\n' + 'bool c6 = a >= b;'
    c_src28 = 'bool c1 = true == true;' + '\n' + 'bool c2 = true == false;' + '\n' + 'bool c3 = false == false;' + '\n' + 'bool c4 = true != true;' + '\n' + 'bool c5 = true != false;' + '\n' + 'bool c6 = false != false;'
    c_src29 = 'bool c1 = true && true;' + '\n' + 'bool c2 = true && false;' + '\n' + 'bool c3 = false && false;' + '\n' + 'bool c4 = true || true;' + '\n' + 'bool c5 = true || false;' + '\n' + 'bool c6 = false || false;'
    c_src30 = 'bool a = false;' + '\n' + 'bool c1 = a && true;' + '\n' + 'bool c2 = false && a;' + '\n' + 'bool c3 = true || a;' + '\n' + 'bool c4 = a || false;'
    c_src31 = 'int a = 1;' + '\n' + 'bool c1 = a && 1;' + '\n' + 'bool c2 = a && 0;' + '\n' + 'bool c3 = a || 1;' + '\n' + 'bool c4 = 0 || a;'
    c_src32 = 'int a = 1, b = 0;' + '\n' + 'bool c = false, d = true;' + '\n' + 'bool c1 = a && b;' + '\n' + 'bool c2 = a && c;' + '\n' + 'bool c3 = c && d;' + '\n' + 'bool c4 = a || b;' + '\n' + 'bool c5 = a || c;' + '\n' + 'bool c6 = c || d;'
    c_src_raise1 = "char a = 'b';"
    c_src_raise2 = 'int a[] = {10, 20};'
    res1 = SymPyExpression(c_src1, 'c').return_expr()
    res2 = SymPyExpression(c_src2, 'c').return_expr()
    res3 = SymPyExpression(c_src3, 'c').return_expr()
    res4 = SymPyExpression(c_src4, 'c').return_expr()
    res5 = SymPyExpression(c_src5, 'c').return_expr()
    res6 = SymPyExpression(c_src6, 'c').return_expr()
    res7 = SymPyExpression(c_src7, 'c').return_expr()
    res8 = SymPyExpression(c_src8, 'c').return_expr()
    res9 = SymPyExpression(c_src9, 'c').return_expr()
    res10 = SymPyExpression(c_src10, 'c').return_expr()
    res11 = SymPyExpression(c_src11, 'c').return_expr()
    res12 = SymPyExpression(c_src12, 'c').return_expr()
    res13 = SymPyExpression(c_src13, 'c').return_expr()
    res14 = SymPyExpression(c_src14, 'c').return_expr()
    res15 = SymPyExpression(c_src15, 'c').return_expr()
    res16 = SymPyExpression(c_src16, 'c').return_expr()
    res17 = SymPyExpression(c_src17, 'c').return_expr()
    res18 = SymPyExpression(c_src18, 'c').return_expr()
    res19 = SymPyExpression(c_src19, 'c').return_expr()
    res20 = SymPyExpression(c_src20, 'c').return_expr()
    res21 = SymPyExpression(c_src21, 'c').return_expr()
    res22 = SymPyExpression(c_src22, 'c').return_expr()
    res23 = SymPyExpression(c_src23, 'c').return_expr()
    res24 = SymPyExpression(c_src24, 'c').return_expr()
    res25 = SymPyExpression(c_src25, 'c').return_expr()
    res26 = SymPyExpression(c_src26, 'c').return_expr()
    res27 = SymPyExpression(c_src27, 'c').return_expr()
    res28 = SymPyExpression(c_src28, 'c').return_expr()
    res29 = SymPyExpression(c_src29, 'c').return_expr()
    res30 = SymPyExpression(c_src30, 'c').return_expr()
    res31 = SymPyExpression(c_src31, 'c').return_expr()
    res32 = SymPyExpression(c_src32, 'c').return_expr()
    assert res1[0] == Declaration(Variable(Symbol('b'), type=IntBaseType(String('intc')), value=Integer(100)))
    assert res1[1] == Declaration(Variable(Symbol('a'), type=IntBaseType(String('intc')), value=Symbol('b')))
    assert res2[0] == Declaration(Variable(Symbol('a'), type=IntBaseType(String('intc')), value=Integer(1)))
    assert res2[1] == Declaration(Variable(Symbol('b'), type=IntBaseType(String('intc')), value=Add(Symbol('a'), Integer(1))))
    assert res3[0] == Declaration(Variable(Symbol('a'), type=FloatType(String('float32'), nbits=Integer(32), nmant=Integer(23), nexp=Integer(8)), value=Float('12.5', precision=53)))
    assert res3[1] == Declaration(Variable(Symbol('b'), type=FloatType(String('float32'), nbits=Integer(32), nmant=Integer(23), nexp=Integer(8)), value=Mul(Float('20.0', precision=53), Symbol('a'))))
    assert res4[0] == Declaration(Variable(Symbol('a'), type=IntBaseType(String('intc')), value=Integer(83)))
    assert res5[0] == Declaration(Variable(Symbol('a'), type=IntBaseType(String('intc')), value=Integer(-4836)))
    assert res6[0] == Declaration(Variable(Symbol('b'), type=IntBaseType(String('intc')), value=Integer(2)))
    assert res6[1] == Declaration(Variable(Symbol('c'), type=IntBaseType(String('intc')), value=Integer(3)))
    assert res6[2] == Declaration(Variable(Symbol('a'), type=IntBaseType(String('intc')), value=Add(Symbol('b'), Mul(Integer(4), Symbol('c')))))
    assert res7[0] == Declaration(Variable(Symbol('b'), type=IntBaseType(String('intc')), value=Integer(1)))
    assert res7[1] == Declaration(Variable(Symbol('c'), type=IntBaseType(String('intc')), value=Add(Symbol('b'), Integer(2))))
    assert res7[2] == Declaration(Variable(Symbol('a'), type=IntBaseType(String('intc')), value=Mul(Integer(10), Pow(Symbol('b'), Integer(2)), Symbol('c'))))
    assert res8[0] == FunctionDefinition(NoneToken(), name=String('func'), parameters=(), body=CodeBlock(Declaration(Variable(Symbol('a'), type=IntBaseType(String('intc')), value=Integer(1))), Declaration(Variable(Symbol('b'), type=IntBaseType(String('intc')), value=Integer(2))), Declaration(Variable(Symbol('temp'), type=IntBaseType(String('intc')), value=Symbol('a'))), Assignment(Variable(Symbol('a')), Symbol('b')), Assignment(Variable(Symbol('b')), Symbol('temp'))))
    assert res9[0] == Declaration(Variable(Symbol('a'), type=IntBaseType(String('intc')), value=Integer(1)))
    assert res9[1] == Declaration(Variable(Symbol('b'), type=IntBaseType(String('intc')), value=Integer(2)))
    assert res9[2] == Declaration(Variable(Symbol('c'), type=IntBaseType(String('intc')), value=Symbol('a')))
    assert res9[3] == Declaration(Variable(Symbol('d'), type=IntBaseType(String('intc')), value=Add(Symbol('a'), Symbol('b'), Symbol('c'))))
    assert res9[4] == Declaration(Variable(Symbol('e'), type=IntBaseType(String('intc')), value=Add(Pow(Symbol('a'), Integer(3)), Mul(Integer(3), Pow(Symbol('a'), Integer(2)), Symbol('b')), Mul(Integer(3), Symbol('a'), Pow(Symbol('b'), Integer(2))), Pow(Symbol('b'), Integer(3)))))
    assert res9[5] == Declaration(Variable(Symbol('f'), type=IntBaseType(String('intc')), value=Mul(Add(Symbol('a'), Symbol('b'), Mul(Integer(-1), Symbol('c'))), Add(Symbol('a'), Symbol('b'), Symbol('c')))))
    assert res9[6] == Declaration(Variable(Symbol('g'), type=IntBaseType(String('intc')), value=Mul(Symbol('a'), Add(Symbol('b'), Mul(Integer(-1), Symbol('c'))), Pow(Add(Symbol('a'), Symbol('b'), Symbol('c'), Symbol('d')), Integer(2)))))
    assert res10[0] == Declaration(Variable(Symbol('a'), type=FloatType(String('float32'), nbits=Integer(32), nmant=Integer(23), nexp=Integer(8)), value=Float('10.0', precision=53)))
    assert res10[1] == Declaration(Variable(Symbol('b'), type=FloatType(String('float32'), nbits=Integer(32), nmant=Integer(23), nexp=Integer(8)), value=Float('2.5', precision=53)))
    assert res10[2] == Declaration(Variable(Symbol('c'), type=FloatType(String('float32'), nbits=Integer(32), nmant=Integer(23), nexp=Integer(8)), value=Add(Pow(Symbol('a'), Integer(2)), Mul(Integer(2), Symbol('a'), Symbol('b')), Pow(Symbol('b'), Integer(2)))))
    assert res11[0] == Declaration(Variable(Symbol('a'), type=FloatType(String('float32'), nbits=Integer(32), nmant=Integer(23), nexp=Integer(8)), value=Float('4.0', precision=53)))
    assert res12[0] == Declaration(Variable(Symbol('a'), type=IntBaseType(String('intc')), value=Integer(25)))
    assert res13[0] == Declaration(Variable(Symbol('a'), type=IntBaseType(String('intc')), value=Integer(-95)))
    assert res14[0] == Declaration(Variable(Symbol('a'), type=IntBaseType(String('intc')), value=Integer(-300)))
    assert res15[0] == Declaration(Variable(Symbol('a'), type=IntBaseType(String('intc')), value=Integer(4)))
    assert res15[1] == Declaration(Variable(Symbol('b'), type=IntBaseType(String('intc')), value=Integer(2)))
    assert res15[2] == Declaration(Variable(Symbol('c'), type=FloatType(String('float32'), nbits=Integer(32), nmant=Integer(23), nexp=Integer(8)), value=Mul(Pow(Symbol('a'), Integer(-1)), Symbol('b'))))
    assert res16[0] == Declaration(Variable(Symbol('a'), type=IntBaseType(String('intc')), value=Integer(2)))
    assert res16[1] == Declaration(Variable(Symbol('d'), type=IntBaseType(String('intc')), value=Integer(5)))
    assert res16[2] == Declaration(Variable(Symbol('n'), type=IntBaseType(String('intc')), value=Integer(10)))
    assert res16[3] == Declaration(Variable(Symbol('s'), type=IntBaseType(String('intc')), value=Mul(Rational(1, 2), Symbol('a'), Add(Mul(Integer(2), Symbol('a')), Mul(Symbol('d'), Add(Symbol('n'), Integer(-1)))))))
    assert res17[0] == Declaration(Variable(Symbol('a'), type=IntBaseType(String('intc')), value=Integer(1)))
    assert res18[0] == Declaration(Variable(Symbol('a'), type=IntBaseType(String('intc')), value=Integer(2)))
    assert res18[1] == Declaration(Variable(Symbol('b'), type=IntBaseType(String('intc')), value=Mod(Symbol('a'), Integer(3))))
    assert res19[0] == Declaration(Variable(Symbol('a'), type=IntBaseType(String('intc')), value=Integer(100)))
    assert res19[1] == Declaration(Variable(Symbol('b'), type=IntBaseType(String('intc')), value=Integer(3)))
    assert res19[2] == Declaration(Variable(Symbol('c'), type=IntBaseType(String('intc')), value=Mod(Symbol('a'), Symbol('b'))))
    assert res20[0] == Declaration(Variable(Symbol('a'), type=IntBaseType(String('intc')), value=Integer(100)))
    assert res20[1] == Declaration(Variable(Symbol('b'), type=IntBaseType(String('intc')), value=Integer(3)))
    assert res20[2] == Declaration(Variable(Symbol('mod'), type=IntBaseType(String('intc')), value=Integer(1000000007)))
    assert res20[3] == Declaration(Variable(Symbol('c'), type=IntBaseType(String('intc')), value=Mod(Add(Symbol('a'), Mul(Integer(100), Pow(Symbol('a'), Integer(-1)), Symbol('b'))), Symbol('mod'))))
    assert res21[0] == Declaration(Variable(Symbol('a'), type=IntBaseType(String('intc')), value=Integer(100)))
    assert res21[1] == Declaration(Variable(Symbol('b'), type=IntBaseType(String('intc')), value=Integer(3)))
    assert res21[2] == Declaration(Variable(Symbol('mod'), type=IntBaseType(String('intc')), value=Integer(1000000007)))
    assert res21[3] == Declaration(Variable(Symbol('c'), type=IntBaseType(String('intc')), value=Mod(Mul(Add(Symbol('a'), Mul(Integer(-1), Symbol('b'))), Add(Symbol('a'), Symbol('b'))), Symbol('mod'))))
    assert res22[0] == Declaration(Variable(Symbol('a'), type=Type(String('bool')), value=false))
    assert res22[1] == Declaration(Variable(Symbol('b'), type=Type(String('bool')), value=true))
    assert res23[0] == Declaration(Variable(Symbol('a'), type=Type(String('bool')), value=true))
    assert res23[1] == Declaration(Variable(Symbol('b'), type=Type(String('bool')), value=true))
    assert res23[2] == Declaration(Variable(Symbol('c'), type=Type(String('bool')), value=false))
    assert res23[3] == Declaration(Variable(Symbol('d'), type=Type(String('bool')), value=false))
    assert res24[0] == Declaration(Variable(Symbol('a'), type=IntBaseType(String('intc')), value=Integer(1)))
    assert res24[1] == Declaration(Variable(Symbol('b'), type=IntBaseType(String('intc')), value=Integer(2)))
    assert res24[2] == Declaration(Variable(Symbol('c1'), type=Type(String('bool')), value=Equality(Symbol('a'), Integer(1))))
    assert res24[3] == Declaration(Variable(Symbol('c2'), type=Type(String('bool')), value=Equality(Symbol('b'), Integer(2))))
    assert res24[4] == Declaration(Variable(Symbol('c3'), type=Type(String('bool')), value=Unequality(Integer(1), Symbol('a'))))
    assert res24[5] == Declaration(Variable(Symbol('c4'), type=Type(String('bool')), value=Unequality(Integer(1), Symbol('b'))))
    assert res24[6] == Declaration(Variable(Symbol('c5'), type=Type(String('bool')), value=StrictLessThan(Symbol('a'), Integer(0))))
    assert res24[7] == Declaration(Variable(Symbol('c6'), type=Type(String('bool')), value=LessThan(Symbol('b'), Integer(10))))
    assert res24[8] == Declaration(Variable(Symbol('c7'), type=Type(String('bool')), value=StrictGreaterThan(Symbol('a'), Integer(0))))
    assert res24[9] == Declaration(Variable(Symbol('c8'), type=Type(String('bool')), value=GreaterThan(Symbol('b'), Integer(11))))
    assert res25[0] == Declaration(Variable(Symbol('a'), type=IntBaseType(String('intc')), value=Integer(3)))
    assert res25[1] == Declaration(Variable(Symbol('b'), type=IntBaseType(String('intc')), value=Integer(4)))
    assert res25[2] == Declaration(Variable(Symbol('c1'), type=Type(String('bool')), value=Equality(Symbol('a'), Symbol('b'))))
    assert res25[3] == Declaration(Variable(Symbol('c2'), type=Type(String('bool')), value=Unequality(Symbol('a'), Symbol('b'))))
    assert res25[4] == Declaration(Variable(Symbol('c3'), type=Type(String('bool')), value=StrictLessThan(Symbol('a'), Symbol('b'))))
    assert res25[5] == Declaration(Variable(Symbol('c4'), type=Type(String('bool')), value=LessThan(Symbol('a'), Symbol('b'))))
    assert res25[6] == Declaration(Variable(Symbol('c5'), type=Type(String('bool')), value=StrictGreaterThan(Symbol('a'), Symbol('b'))))
    assert res25[7] == Declaration(Variable(Symbol('c6'), type=Type(String('bool')), value=GreaterThan(Symbol('a'), Symbol('b'))))
    assert res26[0] == Declaration(Variable(Symbol('a'), type=FloatType(String('float32'), nbits=Integer(32), nmant=Integer(23), nexp=Integer(8)), value=Float('1.25', precision=53)))
    assert res26[1] == Declaration(Variable(Symbol('b'), type=FloatType(String('float32'), nbits=Integer(32), nmant=Integer(23), nexp=Integer(8)), value=Float('2.5', precision=53)))
    assert res26[2] == Declaration(Variable(Symbol('c1'), type=Type(String('bool')), value=Equality(Symbol('a'), Float('1.25', precision=53))))
    assert res26[3] == Declaration(Variable(Symbol('c2'), type=Type(String('bool')), value=Equality(Symbol('b'), Float('2.54', precision=53))))
    assert res26[4] == Declaration(Variable(Symbol('c3'), type=Type(String('bool')), value=Unequality(Float('1.2', precision=53), Symbol('a'))))
    assert res26[5] == Declaration(Variable(Symbol('c4'), type=Type(String('bool')), value=Unequality(Float('1.5', precision=53), Symbol('b'))))
    assert res27[0] == Declaration(Variable(Symbol('a'), type=FloatType(String('float32'), nbits=Integer(32), nmant=Integer(23), nexp=Integer(8)), value=Float('1.25', precision=53)))
    assert res27[1] == Declaration(Variable(Symbol('b'), type=FloatType(String('float32'), nbits=Integer(32), nmant=Integer(23), nexp=Integer(8)), value=Float('2.5', precision=53)))
    assert res27[2] == Declaration(Variable(Symbol('c1'), type=Type(String('bool')), value=Equality(Symbol('a'), Symbol('b'))))
    assert res27[3] == Declaration(Variable(Symbol('c2'), type=Type(String('bool')), value=Unequality(Symbol('a'), Symbol('b'))))
    assert res27[4] == Declaration(Variable(Symbol('c3'), type=Type(String('bool')), value=StrictLessThan(Symbol('a'), Symbol('b'))))
    assert res27[5] == Declaration(Variable(Symbol('c4'), type=Type(String('bool')), value=LessThan(Symbol('a'), Symbol('b'))))
    assert res27[6] == Declaration(Variable(Symbol('c5'), type=Type(String('bool')), value=StrictGreaterThan(Symbol('a'), Symbol('b'))))
    assert res27[7] == Declaration(Variable(Symbol('c6'), type=Type(String('bool')), value=GreaterThan(Symbol('a'), Symbol('b'))))
    assert res28[0] == Declaration(Variable(Symbol('c1'), type=Type(String('bool')), value=true))
    assert res28[1] == Declaration(Variable(Symbol('c2'), type=Type(String('bool')), value=false))
    assert res28[2] == Declaration(Variable(Symbol('c3'), type=Type(String('bool')), value=true))
    assert res28[3] == Declaration(Variable(Symbol('c4'), type=Type(String('bool')), value=false))
    assert res28[4] == Declaration(Variable(Symbol('c5'), type=Type(String('bool')), value=true))
    assert res28[5] == Declaration(Variable(Symbol('c6'), type=Type(String('bool')), value=false))
    assert res29[0] == Declaration(Variable(Symbol('c1'), type=Type(String('bool')), value=true))
    assert res29[1] == Declaration(Variable(Symbol('c2'), type=Type(String('bool')), value=false))
    assert res29[2] == Declaration(Variable(Symbol('c3'), type=Type(String('bool')), value=false))
    assert res29[3] == Declaration(Variable(Symbol('c4'), type=Type(String('bool')), value=true))
    assert res29[4] == Declaration(Variable(Symbol('c5'), type=Type(String('bool')), value=true))
    assert res29[5] == Declaration(Variable(Symbol('c6'), type=Type(String('bool')), value=false))
    assert res30[0] == Declaration(Variable(Symbol('a'), type=Type(String('bool')), value=false))
    assert res30[1] == Declaration(Variable(Symbol('c1'), type=Type(String('bool')), value=Symbol('a')))
    assert res30[2] == Declaration(Variable(Symbol('c2'), type=Type(String('bool')), value=false))
    assert res30[3] == Declaration(Variable(Symbol('c3'), type=Type(String('bool')), value=true))
    assert res30[4] == Declaration(Variable(Symbol('c4'), type=Type(String('bool')), value=Symbol('a')))
    assert res31[0] == Declaration(Variable(Symbol('a'), type=IntBaseType(String('intc')), value=Integer(1)))
    assert res31[1] == Declaration(Variable(Symbol('c1'), type=Type(String('bool')), value=Symbol('a')))
    assert res31[2] == Declaration(Variable(Symbol('c2'), type=Type(String('bool')), value=false))
    assert res31[3] == Declaration(Variable(Symbol('c3'), type=Type(String('bool')), value=true))
    assert res31[4] == Declaration(Variable(Symbol('c4'), type=Type(String('bool')), value=Symbol('a')))
    assert res32[0] == Declaration(Variable(Symbol('a'), type=IntBaseType(String('intc')), value=Integer(1)))
    assert res32[1] == Declaration(Variable(Symbol('b'), type=IntBaseType(String('intc')), value=Integer(0)))
    assert res32[2] == Declaration(Variable(Symbol('c'), type=Type(String('bool')), value=false))
    assert res32[3] == Declaration(Variable(Symbol('d'), type=Type(String('bool')), value=true))
    assert res32[4] == Declaration(Variable(Symbol('c1'), type=Type(String('bool')), value=And(Symbol('a'), Symbol('b'))))
    assert res32[5] == Declaration(Variable(Symbol('c2'), type=Type(String('bool')), value=And(Symbol('a'), Symbol('c'))))
    assert res32[6] == Declaration(Variable(Symbol('c3'), type=Type(String('bool')), value=And(Symbol('c'), Symbol('d'))))
    assert res32[7] == Declaration(Variable(Symbol('c4'), type=Type(String('bool')), value=Or(Symbol('a'), Symbol('b'))))
    assert res32[8] == Declaration(Variable(Symbol('c5'), type=Type(String('bool')), value=Or(Symbol('a'), Symbol('c'))))
    assert res32[9] == Declaration(Variable(Symbol('c6'), type=Type(String('bool')), value=Or(Symbol('c'), Symbol('d'))))
    raises(NotImplementedError, lambda: SymPyExpression(c_src_raise1, 'c'))
    raises(NotImplementedError, lambda: SymPyExpression(c_src_raise2, 'c'))