import pytest
import rpy2.robjects as robjects
import rpy2.rlike.container as rlc
import array
import csv
import os
import tempfile
def test_head():
    dataf = robjects.r('data.frame(a=1:26, b=I(letters))')
    assert dataf.head(5).nrow == 5
    assert dataf.head(5).ncol == 2