import abc
from types import SimpleNamespace
from rpy2.robjects.robject import RObjectMixin
import rpy2.rinterface as rinterface
from rpy2.rinterface import StrSexpVector
from rpy2.robjects import help as rhelp
from rpy2.robjects import conversion
def extends(self):
    """Return the R classes this extends.

        This calls the R function methods::extends()."""
    return methods_env['extends'](self.rclass)