import math
import string
from datetime import datetime, timedelta
from functools import lru_cache
from itertools import cycle
import numpy as np
import pandas as pd
from .utils import find_package_file
@lru_cache()
def generate_date_series():
    if pd.__version__ >= '2.2.0':
        return pd.Series(pd.date_range('1970-01-01', '2099-12-31', freq='D'))
    return pd.Series(pd.date_range('1677-09-23', '2262-04-10', freq='D'))