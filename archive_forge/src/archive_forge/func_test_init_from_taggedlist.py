import pytest
import rpy2.robjects as robjects
import rpy2.rlike.container as rlc
import array
import csv
import os
import tempfile
def test_init_from_taggedlist():
    letters = robjects.r.letters
    numbers = robjects.r('1:26')
    df = robjects.DataFrame(rlc.TaggedList((letters, numbers), tags=('letters', 'numbers')))
    assert df.rclass[0] == 'data.frame'