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
def create_test_dfs(*args, **kwargs):
    post_fn = kwargs.pop('post_fn', lambda df: df)
    return map(post_fn, [pd.DataFrame(*args, **kwargs), pandas.DataFrame(*args, **kwargs)])