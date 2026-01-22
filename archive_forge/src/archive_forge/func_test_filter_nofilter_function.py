import pytest
from rpy2.robjects import packages
from rpy2.robjects import rl
from rpy2.robjects.vectors import StrVector
def test_filter_nofilter_function(self):
    dataf = dplyr.DataFrame(mtcars)
    dataf_filter = dplyr.filter(dataf)
    assert dataf.nrow == dataf_filter.nrow