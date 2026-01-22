import pytest
from rpy2.robjects import packages
from rpy2.robjects import rl
from rpy2.robjects.vectors import StrVector
def test_mutate(self):
    dataf_a = dplyr.DataFrame(mtcars)
    dataf_b = dataf_a.mutate(foo=1, bar=rl('gear+1'))
    assert type(dataf_b) is dplyr.DataFrame
    assert all((a == b for a, b in zip(dataf_a.rx2('gear'), dataf_b.rx2('gear'))))
    assert all((a + 1 == b for a, b in zip(dataf_a.rx2('gear'), dataf_b.rx2('bar'))))