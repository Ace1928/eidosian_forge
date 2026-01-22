import pytest
import sys
from rpy2 import robjects
from rpy2 import rinterface
import rpy2.rlike.container
import rpy2.robjects.conversion as conversion
@pytest.mark.parametrize('cls', (robjects.ListVector, rinterface.ListSexpVector))
def test_rlist_to_numpy(self, cls):
    df = cls(robjects.ListVector({'a': 1, 'b': 2, 'c': robjects.vectors.FactorVector('e')}))
    with (robjects.default_converter + rpyn.converter).context() as cv:
        rec = cv.rpy2py(df)
    assert rpy2.rlike.container.OrdDict == type(rec)
    assert rec['a'][0] == 1
    assert rec['b'][0] == 2
    assert rec['c'][0] == 'e'