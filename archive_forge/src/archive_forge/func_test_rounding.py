import pytest, math, re
import itertools
import operator
from numpy.core._simd import targets, clear_floatstatus, get_floatstatus
from numpy.core._multiarray_umath import __cpu_baseline__
@pytest.mark.parametrize('intrin, func', [('ceil', math.ceil), ('trunc', math.trunc), ('floor', math.floor), ('rint', round)])
def test_rounding(self, intrin, func):
    """
        Test intrinsics:
            npyv_rint_##SFX
            npyv_ceil_##SFX
            npyv_trunc_##SFX
            npyv_floor##SFX
        """
    intrin_name = intrin
    intrin = getattr(self, intrin)
    pinf, ninf, nan = (self._pinfinity(), self._ninfinity(), self._nan())
    round_cases = ((nan, nan), (pinf, pinf), (ninf, ninf))
    for case, desired in round_cases:
        data_round = [desired] * self.nlanes
        _round = intrin(self.setall(case))
        assert _round == pytest.approx(data_round, nan_ok=True)
    for x in range(0, 2 ** 20, 256 ** 2):
        for w in (-1.05, -1.1, -1.15, 1.05, 1.1, 1.15):
            data = self.load([(x + a) * w for a in range(self.nlanes)])
            data_round = [func(x) for x in data]
            _round = intrin(data)
            assert _round == data_round
    for i in (1.1529215045988576e+18, 4.6116860183954304e+18, 5.902958103546122e+20, 2.3611832414184488e+21):
        x = self.setall(i)
        y = intrin(x)
        data_round = [func(n) for n in x]
        assert y == data_round
    if intrin_name == 'floor':
        data_szero = (-0.0,)
    else:
        data_szero = (-0.0, -0.25, -0.3, -0.45, -0.5)
    for w in data_szero:
        _round = self._to_unsigned(intrin(self.setall(w)))
        data_round = self._to_unsigned(self.setall(-0.0))
        assert _round == data_round