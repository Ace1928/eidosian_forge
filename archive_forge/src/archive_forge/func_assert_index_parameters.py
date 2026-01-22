import datetime as dt
from datetime import date
import re
import numpy as np
import pytest
from pandas.compat.numpy import np_long
import pandas as pd
from pandas import (
import pandas._testing as tm
def assert_index_parameters(self, index):
    assert index.freq == '40960ns'
    assert index.inferred_freq == '40960ns'