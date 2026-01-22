import numpy as np
import pandas as pd
from .._utils import get_valid_kwargs, uniquecols
from ..doctools import document
from ..exceptions import PlotnineError
from .stat import stat
def make_summary_fun(fun_data, fun_y, fun_ymin, fun_ymax, fun_args):
    """
    Make summary function
    """
    if isinstance(fun_data, str):
        fun_data = function_dict[fun_data]
    if any([fun_y, fun_ymin, fun_ymax]):

        def func(df) -> pd.DataFrame:
            d = {}
            if fun_y:
                kwargs = get_valid_kwargs(fun_y, fun_args)
                d['y'] = [fun_y(df['y'], **kwargs)]
            if fun_ymin:
                kwargs = get_valid_kwargs(fun_ymin, fun_args)
                d['ymin'] = [fun_ymin(df['y'], **kwargs)]
            if fun_ymax:
                kwargs = get_valid_kwargs(fun_ymax, fun_args)
                d['ymax'] = [fun_ymax(df['y'], **kwargs)]
            return pd.DataFrame(d)
    elif fun_data:
        kwargs = get_valid_kwargs(fun_data, fun_args)

        def func(df) -> pd.DataFrame:
            return fun_data(df['y'], **kwargs)
    else:
        raise ValueError(f'Bad value for function fun_data={fun_data}')
    return func