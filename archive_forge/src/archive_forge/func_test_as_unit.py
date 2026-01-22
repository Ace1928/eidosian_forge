import pytest
from pandas._libs.tslibs.dtypes import NpyDatetimeUnit
from pandas.errors import OutOfBoundsTimedelta
from pandas import Timedelta
def test_as_unit(self):
    td = Timedelta(days=1)
    assert td.as_unit('ns') is td
    res = td.as_unit('us')
    assert res._value == td._value // 1000
    assert res._creso == NpyDatetimeUnit.NPY_FR_us.value
    rt = res.as_unit('ns')
    assert rt._value == td._value
    assert rt._creso == td._creso
    res = td.as_unit('ms')
    assert res._value == td._value // 1000000
    assert res._creso == NpyDatetimeUnit.NPY_FR_ms.value
    rt = res.as_unit('ns')
    assert rt._value == td._value
    assert rt._creso == td._creso
    res = td.as_unit('s')
    assert res._value == td._value // 1000000000
    assert res._creso == NpyDatetimeUnit.NPY_FR_s.value
    rt = res.as_unit('ns')
    assert rt._value == td._value
    assert rt._creso == td._creso