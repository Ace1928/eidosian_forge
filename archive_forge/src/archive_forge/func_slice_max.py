from rpy2 import robjects
from rpy2.robjects.packages import (importr,
import warnings
@result_as
def slice_max(self, *args, **kwargs):
    """Call the R function `dplyr::slice_max()`."""
    res = dplyr.slice_max(self, *args, **kwargs)
    return res