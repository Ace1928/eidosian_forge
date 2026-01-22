import warnings
import sys
import os
import itertools
import pytest
import weakref
import numpy as np
from numpy.testing import (
def make_cycle():
    a = []
    a.append(a)
    a.append(a)
    return a