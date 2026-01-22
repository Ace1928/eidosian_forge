from __future__ import annotations
import glob
import os
import shutil
import tempfile
import numpy as np
import pandas as pd
import pytest
import dask.dataframe as dd
from dask.dataframe.optimize import optimize_dataframe_getitem
from dask.dataframe.utils import assert_eq
def test_to_orc_delayed(tmp_path):
    df = pd.DataFrame(np.random.randn(100, 4), columns=['a', 'b', 'c', 'd'])
    ddf = dd.from_pandas(df, npartitions=4)
    eager_path = os.path.join(tmp_path, 'eager_orc_dataset')
    ddf.to_orc(eager_path)
    assert len(glob.glob(os.path.join(eager_path, '*'))) == 4
    delayed_path = os.path.join(tmp_path, 'delayed_orc_dataset')
    dataset = ddf.to_orc(delayed_path, compute=False)
    dataset.compute()
    assert len(glob.glob(os.path.join(delayed_path, '*'))) == 4