import numpy
import numpy as np
import datetime
import pytest
from numpy.testing import (
from numpy.compat import pickle
def cast2():
    numpy.datetime64('2014').astype('<M8[fs]')