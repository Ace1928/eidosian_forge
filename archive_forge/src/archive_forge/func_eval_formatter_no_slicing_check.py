import os
import math
import random
from pathlib import Path
import pytest
from IPython.utils import text
def eval_formatter_no_slicing_check(f):
    ns = dict(n=12, pi=math.pi, stuff='hello there', os=os)
    s = f.format('{n:x} {pi**2:+f}', **ns)
    assert s == 'c +9.869604'
    s = f.format('{stuff[slice(1,4)]}', **ns)
    assert s == 'ell'
    s = f.format('{a[:]}', a=[1, 2])
    assert s == '[1, 2]'