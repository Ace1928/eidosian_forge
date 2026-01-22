import pythran.metadata as metadata
import pythran.openmp as openmp
from pythran.utils import isnum
import gast as ast
import os
import sys
import io
def write_pair(pair):
    k, v = pair
    self.dispatch(k)
    self.write(': ')
    self.dispatch(v)