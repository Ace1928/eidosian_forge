import collections
import importlib.util
import json
import os
import tempfile
from typing import Any, Callable, Dict, Type
import numpy as np
import xgboost as xgb
from xgboost._typing import ArrayLike
def validate_data_initialization(dmatrix: Type, model: Type[xgb.XGBModel], X: ArrayLike, y: ArrayLike) -> None:
    """Assert that we don't create duplicated DMatrix."""
    old_init = dmatrix.__init__
    count = [0]

    def new_init(self: Any, **kwargs: Any) -> Callable:
        count[0] += 1
        return old_init(self, **kwargs)
    dmatrix.__init__ = new_init
    model(n_estimators=1).fit(X, y, eval_set=[(X, y)])
    assert count[0] == 1
    count[0] = 0
    y_copy = y.copy()
    model(n_estimators=1).fit(X, y, eval_set=[(X, y_copy)])
    assert count[0] == 2
    dmatrix.__init__ = old_init