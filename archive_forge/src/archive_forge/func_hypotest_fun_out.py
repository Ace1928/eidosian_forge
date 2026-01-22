import numpy as np
from functools import wraps
from scipy._lib._docscrape import FunctionDoc, Parameter
from scipy._lib._util import _contains_nan, AxisError, _get_nan
import inspect
def hypotest_fun_out(*samples, **kwds):
    new_kwds = dict(zip(kwd_samp, samples[n_samp:]))
    kwds.update(new_kwds)
    return hypotest_fun_in(*samples[:n_samp], **kwds)