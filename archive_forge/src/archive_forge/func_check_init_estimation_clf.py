import numpy as np
import pandas as pd
from dask import array as da
from dask import dataframe as dd
from distributed import Client
import xgboost as xgb
from xgboost.testing.updater import get_basescore
def check_init_estimation_clf(tree_method: str, client: Client) -> None:
    """Test init estimation for classsifier."""
    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=4096 * 2, n_features=32, random_state=1994)
    clf = xgb.XGBClassifier(n_estimators=1, max_depth=1, tree_method=tree_method)
    clf.fit(X, y)
    base_score = get_basescore(clf)
    dx = da.from_array(X).rechunk(chunks=(32, None))
    dy = da.from_array(y).rechunk(chunks=(32,))
    dclf = xgb.dask.DaskXGBClassifier(n_estimators=1, max_depth=1, tree_method=tree_method)
    dclf.client = client
    dclf.fit(dx, dy)
    dbase_score = get_basescore(dclf)
    np.testing.assert_allclose(base_score, dbase_score)