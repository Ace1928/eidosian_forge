import warnings
import rpy2.rinterface as rinterface
import rpy2.robjects as robjects
import rpy2.robjects.conversion as conversion
from rpy2.robjects.packages import importr, WeakPackage
@classmethod
def yaxisgrob(cls, **kwargs):
    """ Constructor (uses the R function grid::yaxisgrob())"""
    res = cls._yaxisgrob(**kwargs)
    return cls(res)