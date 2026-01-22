from numpy.testing import (
from numpy import (
import numpy as np
import pytest
def get_mat(n):
    data = arange(n)
    data = add.outer(data, data)
    return data