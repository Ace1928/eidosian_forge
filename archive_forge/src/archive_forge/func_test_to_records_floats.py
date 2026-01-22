from collections import abc
import email
from email.parser import Parser
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_to_records_floats(self):
    df = DataFrame(np.random.default_rng(2).random((10, 10)))
    df.to_records()