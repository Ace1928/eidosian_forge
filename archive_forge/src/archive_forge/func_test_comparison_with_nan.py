import pytest, math, re
import itertools
import operator
from numpy.core._simd import targets, clear_floatstatus, get_floatstatus
from numpy.core._multiarray_umath import __cpu_baseline__
@pytest.mark.parametrize('py_comp,np_comp', [(operator.lt, 'cmplt'), (operator.le, 'cmple'), (operator.gt, 'cmpgt'), (operator.ge, 'cmpge'), (operator.eq, 'cmpeq'), (operator.ne, 'cmpneq')])
def test_comparison_with_nan(self, py_comp, np_comp):
    pinf, ninf, nan = (self._pinfinity(), self._ninfinity(), self._nan())
    mask_true = self._true_mask()

    def to_bool(vector):
        return [lane == mask_true for lane in vector]
    intrin = getattr(self, np_comp)
    cmp_cases = ((0, nan), (nan, 0), (nan, nan), (pinf, nan), (ninf, nan), (-0.0, +0.0))
    for case_operand1, case_operand2 in cmp_cases:
        data_a = [case_operand1] * self.nlanes
        data_b = [case_operand2] * self.nlanes
        vdata_a = self.setall(case_operand1)
        vdata_b = self.setall(case_operand2)
        vcmp = to_bool(intrin(vdata_a, vdata_b))
        data_cmp = [py_comp(a, b) for a, b in zip(data_a, data_b)]
        assert vcmp == data_cmp