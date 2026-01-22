import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def float_result_type(dtype, dtype2):
    typs = {dtype.kind, dtype2.kind}
    if not len(typs - {'f', 'i', 'u'}) and (dtype.kind == 'f' or dtype2.kind == 'f'):
        return 'f'
    return None