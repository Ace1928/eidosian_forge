import pytest
from rpy2.robjects import packages
from rpy2.robjects import rl
from rpy2.robjects.vectors import StrVector
def test_mutate_all(self):
    dataf_a = dplyr.DataFrame(mtcars)
    dataf_b = dataf_a.mutate_all(rl('sqrt'))
    assert type(dataf_b) is dplyr.DataFrame