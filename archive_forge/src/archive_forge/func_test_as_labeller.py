import pytest
from rpy2.robjects import packages
from rpy2.robjects import rl
@pytest.mark.parametrize('labeller', (rl('as_labeller(c(`0` = "Zero", `1` = "One"))'), {'0': 'Zero', '1': 'One'}))
def test_as_labeller(self, labeller):
    if isinstance(labeller, dict):
        labeller = ggplot2.dict2rvec(labeller)
    gp = ggplot2.ggplot(mtcars) + ggplot2.facet_wrap(rl('~am'), labeller=ggplot2.as_labeller(labeller))
    assert isinstance(gp, ggplot2.GGPlot)