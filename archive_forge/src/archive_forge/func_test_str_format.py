import random
from mpmath import *
from mpmath.libmp import *
def test_str_format():
    assert to_str(from_float(0.1), 15, strip_zeros=False) == '0.100000000000000'
    assert to_str(from_float(0.0), 15, show_zero_exponent=True) == '0.0e+0'
    assert to_str(from_float(0.0), 0, show_zero_exponent=True) == '.0e+0'
    assert to_str(from_float(0.0), 0, show_zero_exponent=False) == '.0'
    assert to_str(from_float(0.0), 1, show_zero_exponent=True) == '0.0e+0'
    assert to_str(from_float(0.0), 1, show_zero_exponent=False) == '0.0'
    assert to_str(from_float(1.23), 3, show_zero_exponent=True) == '1.23e+0'
    assert to_str(from_float(0.0123456789), 15, strip_zeros=False, min_fixed=0, max_fixed=0) == '1.23456789000000e-2'
    assert to_str(from_float(123.456789), 15, strip_zeros=False, min_fixed=0, max_fixed=0) == '1.23456789000000e+2'
    assert to_str(from_float(212870000000000.0), 15, max_fixed=1000) == '212870000000000.0'
    assert to_str(from_float(2128700000000000.0), 15, max_fixed=1000) == '2128700000000000.0'
    assert to_str(from_float(2.1287e+16), 15, max_fixed=1000) == '21287000000000000.0'
    assert to_str(from_float(2.1287e+30), 15, max_fixed=1000) == '2128700000000000000000000000000.0'