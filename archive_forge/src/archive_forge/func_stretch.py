import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def stretch(row):
    if row['variable'] == 'height':
        row['value'] += 0.5
    return row