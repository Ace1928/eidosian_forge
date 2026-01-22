import pytest
from rpy2.robjects import packages
from rpy2.robjects import rl
from rpy2.robjects.vectors import StrVector
def test_dataframe(self):
    dataf = dplyr.DataFrame(mtcars)
    assert isinstance(dataf, dplyr.DataFrame)