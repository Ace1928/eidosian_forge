import numpy as np
import pandas as pd
import pandas._testing as tm
def f_copy(x):
    x = x.copy()
    x['rank'] = x.val.rank(method='min')
    return x.groupby('cat2')['rank'].min()