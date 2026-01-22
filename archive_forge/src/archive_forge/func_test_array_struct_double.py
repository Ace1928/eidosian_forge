import pytest
import rpy2.rinterface as rinterface
@pytest.mark.skipif(not has_numpy, reason='Package numpy is not installed.')
def test_array_struct_double():
    px = [1.0, -2.0, 3.0]
    x = rinterface.FloatSexpVector(px)
    nx = numpy.asarray(x.memoryview())
    assert nx.dtype.kind == 'f'
    for orig, new in zip(px, nx):
        assert orig == new
    nx[1] = 333.2
    assert x[1] == 333.2