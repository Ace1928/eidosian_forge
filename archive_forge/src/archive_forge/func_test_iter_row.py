import pytest
import rpy2.robjects as robjects
import rpy2.rlike.container as rlc
import array
import csv
import os
import tempfile
def test_iter_row():
    dataf = robjects.r('data.frame(a=1:2, b=I(c("a", "b")))')
    rows = [x for x in dataf.iter_row()]
    assert rows[0][0][0] == 1
    assert rows[1][1][0] == 'b'