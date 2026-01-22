from rpy2 import robjects
from rpy2.robjects.packages import (importr,
import warnings
@result_as
def transmute_all(self, *args, **kwargs):
    """Call the R function `dplyr::transmute_all()`."""
    res = dplyr.transmute_all(self, *args, **kwargs)
    return res