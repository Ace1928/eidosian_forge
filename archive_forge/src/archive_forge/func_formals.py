import inspect
import os
import re
import textwrap
import typing
from typing import Union
import warnings
from collections import OrderedDict
from rpy2.robjects.robject import RObjectMixin
import rpy2.rinterface as rinterface
import rpy2.rinterface_lib.sexp
from rpy2.robjects import help
from rpy2.robjects import conversion
from rpy2.robjects.vectors import Vector
from rpy2.robjects.packages_utils import (default_symbol_r2python,
def formals(self):
    """ Return the signature of the underlying R function
        (as the R function 'formals()' would).
        """
    res = _formals_fixed(self)
    res = conversion.get_conversion().rpy2py(res)
    return res