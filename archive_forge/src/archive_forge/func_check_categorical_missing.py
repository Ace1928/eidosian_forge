import json
from functools import partial, update_wrapper
from typing import Any, Dict, List
import numpy as np
import xgboost as xgb
import xgboost.testing as tm
def check_categorical_missing(rows: int, cols: int, cats: int, device: str, tree_method: str) -> None:
    """Check categorical data with missing values."""
    parameters: Dict[str, Any] = {'tree_method': tree_method, 'device': device}
    cat, label = tm.make_categorical(rows, n_features=cols, n_categories=cats, onehot=False, sparsity=0.5)
    Xy = xgb.DMatrix(cat, label, enable_categorical=True)

    def run(max_cat_to_onehot: int) -> None:
        parameters['max_cat_to_onehot'] = max_cat_to_onehot
        evals_result: Dict[str, Dict] = {}
        booster = xgb.train(parameters, Xy, num_boost_round=16, evals=[(Xy, 'Train')], evals_result=evals_result)
        assert tm.non_increasing(evals_result['Train']['rmse'])
        y_predt = booster.predict(Xy)
        rmse = tm.root_mean_square(label, y_predt)
        np.testing.assert_allclose(rmse, evals_result['Train']['rmse'][-1], rtol=2e-05)
    run(USE_ONEHOT)
    run(USE_PART)