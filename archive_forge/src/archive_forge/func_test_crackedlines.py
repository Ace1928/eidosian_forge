import os
import pytest
import numpy as np
from . import util
from numpy.f2py.crackfortran import crackfortran
def test_crackedlines(self):
    mod = crackfortran(str(self.sources[0]))
    print(mod[0]['vars'])
    assert mod[0]['vars']['mydata']['='] == '0'