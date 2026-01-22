from functools import wraps
from statsmodels.tools.data import _is_using_pandas
from statsmodels.tsa.tsatools import freq_to_period
def pandas_wrapper_predict(func, trim_head=None, trim_tail=None, columns=None, *args, **kwargs):
    raise NotImplementedError