import math
import string
from datetime import datetime, timedelta
from functools import lru_cache
from itertools import cycle
import numpy as np
import pandas as pd
from .utils import find_package_file
def get_indicators():
    """A Pandas DataFrame with a subset of the world bank indicators"""
    return pd.read_csv(find_package_file('samples/indicators.csv'))