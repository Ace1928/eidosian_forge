import numpy as np
import pandas as pd
import pytest
def test_ibis_hist():
    df = pd.DataFrame(dict(x=np.arange(10)))
    table = ibis.memtable(df)
    table.hvplot.hist('x')