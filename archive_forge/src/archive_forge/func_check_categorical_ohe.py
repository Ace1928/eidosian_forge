import json
from functools import partial, update_wrapper
from typing import Any, Dict, List
import numpy as np
import xgboost as xgb
import xgboost.testing as tm
def check_categorical_ohe(rows: int, cols: int, rounds: int, cats: int, device: str, tree_method: str) -> None:
    """Test for one-hot encoding with categorical data."""
    onehot, label = tm.make_categorical(rows, cols, cats, True)
    cat, _ = tm.make_categorical(rows, cols, cats, False)
    by_etl_results: Dict[str, Dict[str, List[float]]] = {}
    by_builtin_results: Dict[str, Dict[str, List[float]]] = {}
    parameters: Dict[str, Any] = {'tree_method': tree_method, 'max_cat_to_onehot': USE_ONEHOT, 'device': device}
    m = xgb.DMatrix(onehot, label, enable_categorical=False)
    xgb.train(parameters, m, num_boost_round=rounds, evals=[(m, 'Train')], evals_result=by_etl_results)
    m = xgb.DMatrix(cat, label, enable_categorical=True)
    xgb.train(parameters, m, num_boost_round=rounds, evals=[(m, 'Train')], evals_result=by_builtin_results)
    np.testing.assert_allclose(np.array(by_etl_results['Train']['rmse']), np.array(by_builtin_results['Train']['rmse']), rtol=0.001)
    assert tm.non_increasing(by_builtin_results['Train']['rmse'])
    by_grouping: Dict[str, Dict[str, List[float]]] = {}
    parameters['max_cat_to_onehot'] = USE_PART
    parameters['reg_lambda'] = 0
    m = xgb.DMatrix(cat, label, enable_categorical=True)
    xgb.train(parameters, m, num_boost_round=rounds, evals=[(m, 'Train')], evals_result=by_grouping)
    rmse_oh = by_builtin_results['Train']['rmse']
    rmse_group = by_grouping['Train']['rmse']
    for a, b in zip(rmse_oh, rmse_group):
        assert a >= b
    parameters['reg_lambda'] = 1.0
    by_grouping = {}
    xgb.train(parameters, m, num_boost_round=32, evals=[(m, 'Train')], evals_result=by_grouping)
    assert tm.non_increasing(by_grouping['Train']['rmse']), by_grouping