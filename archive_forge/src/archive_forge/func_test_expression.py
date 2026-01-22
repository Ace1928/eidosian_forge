import pytest
import rpy2.rinterface as ri
def test_expression():
    expression = ri.baseenv.find('expression')
    e = expression(ri.StrSexpVector(['a']), ri.StrSexpVector(['b']))
    assert e.typeof == ri.RTYPES.EXPRSXP
    y = e[0]
    assert y.typeof == ri.RTYPES.STRSXP