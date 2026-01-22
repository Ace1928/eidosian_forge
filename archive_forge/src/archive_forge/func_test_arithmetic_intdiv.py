import pytest, math, re
import itertools
import operator
from numpy.core._simd import targets, clear_floatstatus, get_floatstatus
from numpy.core._multiarray_umath import __cpu_baseline__
def test_arithmetic_intdiv(self):
    """
        Test integer division intrinsics:
            npyv_divisor_##sfx
            npyv_divc_##sfx
        """
    if self._is_fp():
        return
    int_min = self._int_min()

    def trunc_div(a, d):
        """
            Divide towards zero works with large integers > 2^53,
            and wrap around overflow similar to what C does.
            """
        if d == -1 and a == int_min:
            return a
        sign_a, sign_d = (a < 0, d < 0)
        if a == 0 or sign_a == sign_d:
            return a // d
        return (a + sign_d - sign_a) // d + 1
    data = [1, -int_min]
    data += range(0, 2 ** 8, 2 ** 5)
    data += range(0, 2 ** 8, 2 ** 5 - 1)
    bsize = self._scalar_size()
    if bsize > 8:
        data += range(2 ** 8, 2 ** 16, 2 ** 13)
        data += range(2 ** 8, 2 ** 16, 2 ** 13 - 1)
    if bsize > 16:
        data += range(2 ** 16, 2 ** 32, 2 ** 29)
        data += range(2 ** 16, 2 ** 32, 2 ** 29 - 1)
    if bsize > 32:
        data += range(2 ** 32, 2 ** 64, 2 ** 61)
        data += range(2 ** 32, 2 ** 64, 2 ** 61 - 1)
    data += [-x for x in data]
    for dividend, divisor in itertools.product(data, data):
        divisor = self.setall(divisor)[0]
        if divisor == 0:
            continue
        dividend = self.load(self._data(dividend))
        data_divc = [trunc_div(a, divisor) for a in dividend]
        divisor_parms = self.divisor(divisor)
        divc = self.divc(dividend, divisor_parms)
        assert divc == data_divc