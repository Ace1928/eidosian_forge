from rpy2 import robjects
from rpy2.robjects.packages import (importr,
import warnings
@result_as
def full_join(self, *args, **kwargs):
    """Call the R function `dplyr::full_join()`."""
    res = dplyr.full_join(self, *args, **kwargs)
    return res