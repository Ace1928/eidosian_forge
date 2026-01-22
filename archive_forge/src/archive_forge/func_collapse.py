from rpy2 import robjects
from rpy2.robjects.packages import (importr,
import warnings
@result_as
def collapse(self, *args, **kwargs):
    """
        Call the function `collapse` in the R package `dplyr`.
        """
    return dplyr.collapse(self, *args, **kwargs)