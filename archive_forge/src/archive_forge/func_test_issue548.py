import random
from mpmath import *
from mpmath.libmp import *
def test_issue548():
    try:
        mpmathify('(' + '1' * 5000 + '!j')
    except:
        return
    assert False