from rpy2 import robjects
from rpy2.robjects.packages import (importr,
import warnings
@result_as
def sample_frac(self, *args):
    """Call the R function `dplyr::sample_frac()`."""
    res = dplyr.sample_frac(self, *args)
    return res