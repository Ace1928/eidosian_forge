import pytest, math, re
import itertools
import operator
from numpy.core._simd import targets, clear_floatstatus, get_floatstatus
from numpy.core._multiarray_umath import __cpu_baseline__
@pytest.mark.parametrize('intrin', ['max', 'maxp', 'maxn', 'min', 'minp', 'minn'])
def test_max_min(self, intrin):
    """
        Test intrinsics:
            npyv_max_##sfx
            npyv_maxp_##sfx
            npyv_maxn_##sfx
            npyv_min_##sfx
            npyv_minp_##sfx
            npyv_minn_##sfx
            npyv_reduce_max_##sfx
            npyv_reduce_maxp_##sfx
            npyv_reduce_maxn_##sfx
            npyv_reduce_min_##sfx
            npyv_reduce_minp_##sfx
            npyv_reduce_minn_##sfx
        """
    pinf, ninf, nan = (self._pinfinity(), self._ninfinity(), self._nan())
    chk_nan = {'xp': 1, 'np': 1, 'nn': 2, 'xn': 2}.get(intrin[-2:], 0)
    func = eval(intrin[:3])
    reduce_intrin = getattr(self, 'reduce_' + intrin)
    intrin = getattr(self, intrin)
    hf_nlanes = self.nlanes // 2
    cases = (([0.0, -0.0], [-0.0, 0.0]), ([10, -10], [10, -10]), ([pinf, 10], [10, ninf]), ([10, pinf], [ninf, 10]), ([10, -10], [10, -10]), ([-10, 10], [-10, 10]))
    for op1, op2 in cases:
        vdata_a = self.load(op1 * hf_nlanes)
        vdata_b = self.load(op2 * hf_nlanes)
        data = func(vdata_a, vdata_b)
        simd = intrin(vdata_a, vdata_b)
        assert simd == data
        data = func(vdata_a)
        simd = reduce_intrin(vdata_a)
        assert simd == data
    if not chk_nan:
        return
    if chk_nan == 1:
        test_nan = lambda a, b: b if math.isnan(a) else a if math.isnan(b) else b
    else:
        test_nan = lambda a, b: nan if math.isnan(a) or math.isnan(b) else b
    cases = ((nan, 10), (10, nan), (nan, pinf), (pinf, nan), (nan, nan))
    for op1, op2 in cases:
        vdata_ab = self.load([op1, op2] * hf_nlanes)
        data = test_nan(op1, op2)
        simd = reduce_intrin(vdata_ab)
        assert simd == pytest.approx(data, nan_ok=True)
        vdata_a = self.setall(op1)
        vdata_b = self.setall(op2)
        data = [data] * self.nlanes
        simd = intrin(vdata_a, vdata_b)
        assert simd == pytest.approx(data, nan_ok=True)