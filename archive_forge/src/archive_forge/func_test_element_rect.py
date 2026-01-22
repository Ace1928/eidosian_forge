import pytest
from rpy2.robjects import packages
from rpy2.robjects import rl
def test_element_rect(self):
    er = ggplot2.element_rect()
    assert isinstance(er, ggplot2.ElementRect)