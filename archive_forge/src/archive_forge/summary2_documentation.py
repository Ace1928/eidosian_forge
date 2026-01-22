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
Generate LaTeX Summary Table

        Parameters
        ----------
        label : str
            Label of the summary table that can be referenced
            in a latex document (optional)
        