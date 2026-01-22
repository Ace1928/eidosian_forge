import csv
import functools
import itertools
import math
import os
import re
from io import BytesIO
from pathlib import Path
from string import ascii_letters
from typing import Union
import numpy as np
import pandas
import psutil
import pytest
from pandas.core.dtypes.common import (
import modin.pandas as pd
from modin.config import (
from modin.pandas.io import to_pandas
from modin.pandas.testing import (
from modin.utils import try_cast_to_pandas
def generate_multiindex_dfs(axis=1):

    def generate_multiindex(index):
        return pandas.MultiIndex.from_tuples([('a', x) for x in index.values], names=['name1', 'name2'])
    df1, df2 = generate_dfs()
    df1.axes[axis], df2.axes[axis] = map(generate_multiindex, [df1.axes[axis], df2.axes[axis]])
    return (df1, df2)