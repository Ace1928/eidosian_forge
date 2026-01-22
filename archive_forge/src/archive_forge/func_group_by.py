from rpy2 import robjects
from rpy2.robjects.packages import (importr,
import warnings
def group_by(self, *args, _add=False, _drop=robjects.rl('group_by_drop_default(.data)')):
    """Call the R function `dplyr::group_by()`."""
    res = dplyr.group_by(self, *args, **{'.add': _add, '.drop': _drop})
    return GroupedDataFrame(res)