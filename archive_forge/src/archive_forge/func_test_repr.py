import pytest
import rpy2.robjects as robjects
import rpy2.rlike.container as rlc
import array
import csv
import os
import tempfile
def test_repr():
    dataf = robjects.r('data.frame(a=1:2, b=I(c("a", "b")))')
    s = repr(dataf)
    assert 'data.frame' in s.split(os.linesep)[1]