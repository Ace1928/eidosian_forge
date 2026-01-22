from __future__ import division
import uncertainties
import uncertainties.core as uncert_core
from uncertainties import ufloat, unumpy, test_uncertainties
from uncertainties.unumpy import core
from uncertainties.test_uncertainties import numbers_close, arrays_close
def test_wrap_array_func():
    """
    Test of numpy.wrap_array_func(), with optional arguments and
    keyword arguments.
    """

    def f_unc(mat, *args, **kwargs):
        return mat.I + args[0] * kwargs['factor']

    def f(mat, *args, **kwargs):
        assert not any((isinstance(v, uncert_core.UFloat) for v in mat.flat))
        return f_unc(mat, *args, **kwargs)
    f_wrapped = core.wrap_array_func(f)
    m = unumpy.matrix([[ufloat(10, 1), -3.1], [0, ufloat(3, 0)], [1, -3.1]])
    m_f_wrapped = f_wrapped(m, 2, factor=10)
    m_f_unc = f_unc(m, 2, factor=10)
    assert arrays_close(m_f_wrapped, m_f_unc)