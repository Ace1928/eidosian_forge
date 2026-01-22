from __future__ import annotations
import numpy as np
import pandas as pd
from pandas.util import hash_pandas_object
def reduce_state(Ms, b):
    m = 1 << b
    Ms = Ms.reshape(len(Ms) // m, m)
    return Ms.max(axis=0)