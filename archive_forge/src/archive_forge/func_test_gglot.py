import pytest
from rpy2.robjects import packages
from rpy2.robjects import rl
def test_gglot(self):
    gp = ggplot2.ggplot(mtcars)
    assert isinstance(gp, ggplot2.GGPlot)