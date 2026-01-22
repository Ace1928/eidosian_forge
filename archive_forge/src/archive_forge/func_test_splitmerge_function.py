import pytest
from rpy2.robjects import packages
from rpy2.robjects import rl
from rpy2.robjects.vectors import StrVector
def test_splitmerge_function(self):
    dataf = dplyr.DataFrame(mtcars)
    dataf_by_gear = dataf.group_by(rl('gear'))
    dataf_avg_mpg = dataf_by_gear.summarize(foo=rl('mean(mpg)'))
    assert isinstance(dataf_avg_mpg, dplyr.DataFrame)