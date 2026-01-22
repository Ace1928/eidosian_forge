from rpy2 import robjects
from rpy2.robjects.packages import (importr,
import warnings
@result_as
def slice_tail(self, *args, **kwargs):
    """Call the R function `dplyr::slice_tail()`."""
    res = dplyr.slice_tail(self, *args, **kwargs)
    return res