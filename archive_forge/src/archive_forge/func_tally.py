from rpy2 import robjects
from rpy2.robjects.packages import (importr,
import warnings
@result_as
def tally(self, *args, **kwargs):
    """Call the R function `dplyr::transmute()`."""
    res = dplyr.tally(self, *args, **kwargs)
    return res