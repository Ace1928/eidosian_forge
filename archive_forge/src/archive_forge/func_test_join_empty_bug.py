import re
import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_join_empty_bug(self):
    x = DataFrame()
    x.join(DataFrame([3], index=[0], columns=['A']), how='outer')