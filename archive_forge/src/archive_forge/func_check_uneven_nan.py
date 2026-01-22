import numpy as np
import pandas as pd
from dask import array as da
from dask import dataframe as dd
from distributed import Client
import xgboost as xgb
from xgboost.testing.updater import get_basescore
def check_uneven_nan(client: Client, tree_method: str, n_workers: int) -> None:
    """Issue #9271, not every worker has missing value."""
    assert n_workers >= 2
    with client.as_current():
        clf = xgb.dask.DaskXGBClassifier(tree_method=tree_method)
        X = pd.DataFrame({'a': range(10000), 'b': range(10000, 0, -1)})
        y = pd.Series([*[0] * 5000, *[1] * 5000])
        X['a'][:3000:1000] = np.NaN
        client.wait_for_workers(n_workers=n_workers)
        clf.fit(dd.from_pandas(X, npartitions=n_workers), dd.from_pandas(y, npartitions=n_workers))