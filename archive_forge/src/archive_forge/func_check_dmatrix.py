import numpy as np
import pandas
import pytest
import xgboost as xgb
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score
import modin.experimental.xgboost as mxgb
import modin.pandas as pd
from modin.config import Engine
from modin.utils import try_cast_to_pandas
def check_dmatrix(data, label=None, **kwargs):
    modin_data = pd.DataFrame(data)
    modin_label = label if label is None else pd.Series(label)
    try:
        dm = xgb.DMatrix(data, label=label, **kwargs)
    except Exception as xgb_exception:
        with pytest.raises(Exception) as mxgb_exception:
            mxgb.DMatrix(modin_data, label=modin_label, **kwargs)
        assert isinstance(xgb_exception, type(mxgb_exception.value)), 'Got Modin Exception type {}, but xgboost Exception type {} was expected'.format(type(mxgb_exception.value), type(xgb_exception))
    else:
        md_dm = mxgb.DMatrix(modin_data, label=modin_label, **kwargs)
        assert md_dm.num_row() == dm.num_row()
        assert md_dm.num_col() == dm.num_col()
        assert md_dm.feature_names == dm.feature_names
        assert md_dm.feature_types == dm.feature_types