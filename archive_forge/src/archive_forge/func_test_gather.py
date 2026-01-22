import pytest
from rpy2.robjects import packages
from rpy2 import rinterface
from rpy2.robjects import vectors
def test_gather(self):
    dataf = tidyr.DataFrame({'a': 1.0, 'b': 2.0})
    dataf_gathered = dataf.gather('label', 'x')
    assert sorted(['label', 'x']) == sorted(list(dataf_gathered.colnames))