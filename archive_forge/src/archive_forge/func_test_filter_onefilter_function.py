import pytest
from rpy2.robjects import packages
from rpy2.robjects import rl
from rpy2.robjects.vectors import StrVector
def test_filter_onefilter_function(self):
    dataf = dplyr.DataFrame(mtcars)
    ngear_gt_3 = len(tuple((x for x in dataf.rx2('gear') if x > 3)))
    dataf_filter = dplyr.filter(dataf, rl('gear > 3'))
    assert ngear_gt_3 == dataf_filter.nrow