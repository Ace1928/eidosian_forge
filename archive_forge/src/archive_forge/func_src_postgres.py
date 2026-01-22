from rpy2 import robjects
from rpy2.robjects.packages import (importr,
import warnings
def src_postgres(*args, **kwargs):
    res = dplyr.src_postgres(*args, **kwargs)
    return DataSource(res)