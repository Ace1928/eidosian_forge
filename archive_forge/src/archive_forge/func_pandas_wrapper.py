from functools import wraps
from statsmodels.tools.data import _is_using_pandas
from statsmodels.tsa.tsatools import freq_to_period
def pandas_wrapper(func, trim_head=None, trim_tail=None, names=None, *args, **kwargs):

    @wraps(func)
    def new_func(X, *args, **kwargs):
        if not _is_using_pandas(X, None):
            return func(X, *args, **kwargs)
        wrapper_func = _get_pandas_wrapper(X, trim_head, trim_tail, names)
        ret = func(X, *args, **kwargs)
        ret = wrapper_func(ret)
        return ret
    return new_func