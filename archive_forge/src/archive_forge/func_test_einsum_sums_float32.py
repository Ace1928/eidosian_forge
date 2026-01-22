import itertools
import sys
import platform
import pytest
import numpy as np
from numpy.testing import (
def test_einsum_sums_float32(self):
    self.check_einsum_sums('f4')