import math
from sympy.core.containers import Tuple
from sympy.core.numbers import nan, oo, Float, Integer
from sympy.core.relational import Lt
from sympy.core.symbol import symbols, Symbol
from sympy.functions.elementary.trigonometric import sin
from sympy.matrices.dense import Matrix
from sympy.matrices.expressions.matexpr import MatrixSymbol
from sympy.sets.fancysets import Range
from sympy.tensor.indexed import Idx, IndexedBase
from sympy.testing.pytest import raises
from sympy.codegen.ast import (
def test_FloatType():
    assert f16.dig == 3
    assert f32.dig == 6
    assert f64.dig == 15
    assert f80.dig == 18
    assert f128.dig == 33
    assert f16.decimal_dig == 5
    assert f32.decimal_dig == 9
    assert f64.decimal_dig == 17
    assert f80.decimal_dig == 21
    assert f128.decimal_dig == 36
    assert f16.max_exponent == 16
    assert f32.max_exponent == 128
    assert f64.max_exponent == 1024
    assert f80.max_exponent == 16384
    assert f128.max_exponent == 16384
    assert f16.min_exponent == -13
    assert f32.min_exponent == -125
    assert f64.min_exponent == -1021
    assert f80.min_exponent == -16381
    assert f128.min_exponent == -16381
    assert abs(f16.eps / Float('0.00097656', precision=16) - 1) < 0.1 * 10 ** (-f16.dig)
    assert abs(f32.eps / Float('1.1920929e-07', precision=32) - 1) < 0.1 * 10 ** (-f32.dig)
    assert abs(f64.eps / Float('2.2204460492503131e-16', precision=64) - 1) < 0.1 * 10 ** (-f64.dig)
    assert abs(f80.eps / Float('1.08420217248550443401e-19', precision=80) - 1) < 0.1 * 10 ** (-f80.dig)
    assert abs(f128.eps / Float(' 1.92592994438723585305597794258492732e-34', precision=128) - 1) < 0.1 * 10 ** (-f128.dig)
    assert abs(f16.max / Float('65504', precision=16) - 1) < 0.1 * 10 ** (-f16.dig)
    assert abs(f32.max / Float('3.40282347e+38', precision=32) - 1) < 0.1 * 10 ** (-f32.dig)
    assert abs(f64.max / Float('1.79769313486231571e+308', precision=64) - 1) < 0.1 * 10 ** (-f64.dig)
    assert abs(f80.max / Float('1.18973149535723176502e+4932', precision=80) - 1) < 0.1 * 10 ** (-f80.dig)
    assert abs(f128.max / Float('1.18973149535723176508575932662800702e+4932', precision=128) - 1) < 0.1 * 10 ** (-f128.dig)
    assert abs(f16.tiny / Float('6.1035e-05', precision=16) - 1) < 0.1 * 10 ** (-f16.dig)
    assert abs(f32.tiny / Float('1.17549435e-38', precision=32) - 1) < 0.1 * 10 ** (-f32.dig)
    assert abs(f64.tiny / Float('2.22507385850720138e-308', precision=64) - 1) < 0.1 * 10 ** (-f64.dig)
    assert abs(f80.tiny / Float('3.36210314311209350626e-4932', precision=80) - 1) < 0.1 * 10 ** (-f80.dig)
    assert abs(f128.tiny / Float('3.3621031431120935062626778173217526e-4932', precision=128) - 1) < 0.1 * 10 ** (-f128.dig)
    assert f64.cast_check(0.5) == Float(0.5, 17)
    assert abs(f64.cast_check(3.7) - 3.7) < 3e-17
    assert isinstance(f64.cast_check(3), (Float, float))
    assert f64.cast_nocheck(oo) == float('inf')
    assert f64.cast_nocheck(-oo) == float('-inf')
    assert f64.cast_nocheck(float(oo)) == float('inf')
    assert f64.cast_nocheck(float(-oo)) == float('-inf')
    assert math.isnan(f64.cast_nocheck(nan))
    assert f32 != f64
    assert f64 == f64.func(*f64.args)