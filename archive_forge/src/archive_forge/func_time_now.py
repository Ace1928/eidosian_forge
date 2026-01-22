from statsmodels.compat.pandas import FUTURE_STACK
from statsmodels.compat.python import lzip
import datetime
from functools import reduce
import re
import textwrap
import numpy as np
import pandas as pd
from .table import SimpleTable
from .tableformatting import fmt_latex, fmt_txt
def time_now(*args, **kwds):
    now = datetime.datetime.now()
    return now.strftime('%Y-%m-%d %H:%M')