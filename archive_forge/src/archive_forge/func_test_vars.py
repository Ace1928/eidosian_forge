import pytest
from rpy2.robjects import packages
from rpy2.robjects import rl
def test_vars(self):
    gp = ggplot2.ggplot(mtcars) + ggplot2.aes(x='wt', y='mpg') + ggplot2.geom_point() + ggplot2.facet_wrap(ggplot2.vars('gears'))
    assert isinstance(gp, ggplot2.GGPlot)