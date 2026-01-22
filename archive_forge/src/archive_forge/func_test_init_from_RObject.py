import pytest
import rpy2.robjects as robjects
import rpy2.rlike.container as rlc
import array
import csv
import os
import tempfile
def test_init_from_RObject():
    numbers = robjects.r('1:5')
    dataf = robjects.DataFrame(numbers)
    assert len(dataf) == 5
    assert all((len(x) == 1 for x in dataf))
    rfunc = robjects.r('sum')
    with pytest.raises(ValueError):
        robjects.DataFrame(rfunc)
    rdataf = robjects.r('data.frame(a=1:2, b=c("a", "b"))')
    dataf = robjects.DataFrame(rdataf)