import abc
from types import SimpleNamespace
from rpy2.robjects.robject import RObjectMixin
import rpy2.rinterface as rinterface
from rpy2.rinterface import StrSexpVector
from rpy2.robjects import help as rhelp
from rpy2.robjects import conversion
@staticmethod
def isclass(name):
    """ Return whether the given name is a defined class. """
    name = conversion.get_conversion().py2rpy(name)
    return methods_env['isClass'](name)[0]