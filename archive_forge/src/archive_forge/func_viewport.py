import warnings
import rpy2.rinterface as rinterface
import rpy2.robjects as robjects
import rpy2.robjects.conversion as conversion
from rpy2.robjects.packages import importr, WeakPackage
@classmethod
def viewport(cls, **kwargs):
    """ Constructor: create a Viewport """
    res = cls._viewport(**kwargs)
    res = cls(res)
    return res