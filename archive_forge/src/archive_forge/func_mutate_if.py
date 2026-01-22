from rpy2 import robjects
from rpy2.robjects.packages import (importr,
import warnings
@result_as
def mutate_if(self, *args, **kwargs):
    """Call the R function `dplyr::mutate_if()`."""
    res = dplyr.mutate_if(self, *args, **kwargs)
    return res