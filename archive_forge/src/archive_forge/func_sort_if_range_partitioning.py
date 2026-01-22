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
def sort_if_range_partitioning(df1, df2, comparator=None):
    """Sort the passed objects if 'RangePartitioning' is enabled and compare the sorted results."""
    if comparator is None:
        comparator = df_equals
    if RangePartitioning.get() or use_range_partitioning_groupby():
        df1, df2 = (sort_data(df1), sort_data(df2))
    comparator(df1, df2)