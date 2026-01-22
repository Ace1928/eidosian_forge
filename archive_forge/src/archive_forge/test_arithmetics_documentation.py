import operator
import numpy as np
import pytest
import pandas as pd
from pandas import SparseDtype
import pandas._testing as tm
from pandas.core.arrays.sparse import SparseArray

    Fixture returning True or False, determining whether to operate
    op(sparse, dense) instead of op(sparse, sparse)
    