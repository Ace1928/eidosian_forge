import abc
from types import SimpleNamespace
from rpy2.robjects.robject import RObjectMixin
import rpy2.rinterface as rinterface
from rpy2.rinterface import StrSexpVector
from rpy2.robjects import help as rhelp
from rpy2.robjects import conversion
def validobject(self, test=False, complete=False):
    """ Return whether the instance is 'valid' for its class. """
    cv = conversion.get_conversion()
    test = cv.py2rpy(test)
    complete = cv.py2rpy(complete)
    return methods_env['validObject'](self, test=test, complete=complete)[0]