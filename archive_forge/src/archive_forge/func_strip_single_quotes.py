import itertools
import re
from collections import OrderedDict
from collections.abc import Generator
from typing import List
import numpy as np
import scipy as sp
from ..externals import _arff
from ..externals._arff import ArffSparseDataType
from ..utils import (
from ..utils.fixes import pd_fillna
def strip_single_quotes(input_string):
    match = re.search(single_quote_pattern, input_string)
    if match is None:
        return input_string
    return match.group('contents')