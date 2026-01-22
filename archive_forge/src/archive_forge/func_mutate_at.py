from rpy2 import robjects
from rpy2.robjects.packages import (importr,
import warnings
@result_as
def mutate_at(self, *args, **kwargs):
    """Call the R function `dplyr::mutate_at()`."""
    res = dplyr.mutate_at(self, *args, **kwargs)
    return res