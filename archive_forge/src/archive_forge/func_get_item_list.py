import pytest
import rpy2.robjects as robjects
import os
import array
import time
import datetime
import rpy2.rlike.container as rlc
from collections import OrderedDict
def get_item_list():
    mylist = rlist(letters, 'foo')
    idem = robjects.baseenv['identical']
    assert idem(letters, mylist[0])[0] is True
    assert idem('foo', mylist[1])[0] is True