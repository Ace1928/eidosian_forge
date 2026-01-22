from rpy2 import robjects
from rpy2.robjects.packages import (importr,
import warnings
@result_as
def filter(self, *args, _preserve=False):
    """Call the R function `dplyr::filter()`."""
    res = dplyr.filter(self, *args, **{'.preserve': _preserve})
    return res