import string
from datetime import datetime
import numpy as np
import pandas as pd
def getSeriesData():
    index = makeStringIndex(_N)
    return {c: pd.Series(np.random.default_rng(i).standard_normal(_N), index=index) for i, c in enumerate(getCols(_K))}