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
def rcall(self, keyvals, environment: typing.Optional[rinterface.SexpEnvironment]=None) -> rinterface.sexp.Sexp:
    """ Wrapper around the parent method
        rpy2.rinterface.SexpClosure.rcall(). """
    res = super(Function, self).rcall(keyvals, environment=environment)
    return res