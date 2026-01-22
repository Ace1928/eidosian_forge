import pytest
from rpy2.robjects import packages
from rpy2 import rinterface
from rpy2.robjects import vectors
def test_spread(self):
    labels = ('a', 'b', 'c', 'd', 'e')
    dataf = tidyr.DataFrame({'x': vectors.IntVector((1, 2, 3, 4, 5)), 'labels': vectors.StrVector(labels)})
    dataf_spread = dataf.spread('labels', 'x')
    assert sorted(list(labels)) == sorted(list(dataf_spread.colnames))