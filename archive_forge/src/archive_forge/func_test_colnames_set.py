import pytest
import rpy2.robjects as robjects
import rpy2.rlike.container as rlc
import array
import csv
import os
import tempfile
def test_colnames_set():
    dataf = robjects.r('data.frame(a=1:2, b=I(c("a", "b")))')
    dataf.colnames = robjects.StrVector('de')
    assert tuple(dataf.colnames) == ('d', 'e')