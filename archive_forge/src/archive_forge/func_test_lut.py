import pytest, math, re
import itertools
import operator
from numpy.core._simd import targets, clear_floatstatus, get_floatstatus
from numpy.core._multiarray_umath import __cpu_baseline__
@pytest.mark.parametrize('intrin, table_size, elsize', [('self.lut32', 32, 32), ('self.lut16', 16, 64)])
def test_lut(self, intrin, table_size, elsize):
    """
        Test lookup table intrinsics:
            npyv_lut32_##sfx
            npyv_lut16_##sfx
        """
    if elsize != self._scalar_size():
        return
    intrin = eval(intrin)
    idx_itrin = getattr(self.npyv, f'setall_u{elsize}')
    table = range(0, table_size)
    for i in table:
        broadi = self.setall(i)
        idx = idx_itrin(i)
        lut = intrin(table, idx)
        assert lut == broadi