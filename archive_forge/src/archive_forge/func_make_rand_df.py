from typing import Any, Dict, List, Tuple
from unittest import TestCase
import numpy as np
import pandas as pd
from pandasql import sqldf
from datetime import datetime, timedelta
from qpd.dataframe import DataFrame, Column
from qpd import run_sql
from qpd.qpd_engine import QPDEngine
from qpd_test.utils import assert_df_eq
def make_rand_df(self, size: int, **kwargs: Any) -> Tuple[Any, List[str]]:
    np.random.seed(0)
    data: Dict[str, np.ndarray] = {}
    for k, v in kwargs.items():
        if not isinstance(v, tuple):
            v = (v, 0.0)
        dt, null_ct = (v[0], v[1])
        if dt is int:
            s = np.random.randint(10, size=size)
        elif dt is bool:
            s = np.where(np.random.randint(2, size=size), True, False)
        elif dt is float:
            s = np.random.rand(size)
        elif dt is str:
            r = [f'ssssss{x}' for x in range(10)]
            c = np.random.randint(10, size=size)
            s = np.array([r[x] for x in c])
        elif dt is datetime:
            rt = [datetime(2020, 1, 1) + timedelta(days=x) for x in range(10)]
            c = np.random.randint(10, size=size)
            s = np.array([rt[x] for x in c])
        else:
            raise NotImplementedError
        ps = pd.Series(s)
        if null_ct > 0:
            idx = np.random.choice(size, null_ct, replace=False).tolist()
            ps[idx] = None
        data[k] = ps
    pdf = pd.DataFrame(data)
    return (pdf.values.tolist(), list(pdf.columns))