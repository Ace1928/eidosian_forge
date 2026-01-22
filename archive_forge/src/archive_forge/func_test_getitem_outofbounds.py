import pytest
import rpy2.robjects as robjects
import os
import array
import time
import datetime
import rpy2.rlike.container as rlc
from collections import OrderedDict
def test_getitem_outofbounds():
    letters = robjects.baseenv['letters']
    with pytest.raises(IndexError):
        letters[26]