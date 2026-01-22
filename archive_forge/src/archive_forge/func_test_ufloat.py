from builtins import str
import sys
import os
def test_ufloat():
    """
        Test of the transformation of ufloat(tuple,...) and
        ufloat(string,...) into ufloat(nominal_value, std_dev, tag=...).
        """
    tests = {'ufloat((3, 0.14))': 'ufloat(3, 0.14)', 'ufloat((3, 0.14), "pi")': 'ufloat(3, 0.14, "pi")', "ufloat((3, 0.14), 'pi')": "ufloat(3, 0.14, 'pi')", "x = ufloat((3, 0.14), tag='pi')": "x = ufloat(3, 0.14, tag='pi')", 'ufloat((n, s), tag="var")': 'ufloat(n, s, tag="var")', 'ufloat(str_repr, tag="var")': 'ufloat(str_repr, tag="var")', 'ufloat(*tuple_repr, tag="var")': 'ufloat(*tuple_repr, tag="var")', 'ufloat(*t[0, 0])': 'ufloat(*t[0, 0])', 'ufloat("-1.23(3.4)")': 'ufloat_fromstr("-1.23(3.4)")', "ufloat('-1.23(3.4)')": "ufloat_fromstr('-1.23(3.4)')", 'ufloat("-1.23(3.4)", "var")': 'ufloat_fromstr("-1.23(3.4)", "var")', 'ufloat("-1.23(3.4)", tag="var")': 'ufloat_fromstr("-1.23(3.4)", tag="var")'}
    tests.update(dict(((orig.replace('ufloat', 'unc.ufloat'), new.replace('ufloat', 'unc.ufloat')) for orig, new in tests.items())))
    tests[' t  =  u.ufloat("3")'] = ' t  =  u.ufloat_fromstr("3")'
    tests.update(dict(((orig + '**2', new + '**2') for orig, new in tests.items())))
    tests['2**ufloat("3")'] = '2**ufloat_fromstr("3")'
    tests['-ufloat("3")'] = '-ufloat_fromstr("3")'
    check_all('ufloat', tests)