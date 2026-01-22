from rpy2 import robjects
from rpy2.robjects.packages import (importr,
import warnings
def inner(self, *args, **kwargs):
    if constructor is None:
        wrap = type(self)
    else:
        wrap = constructor
    res = func(self, *args, **kwargs)
    return wrap(res)