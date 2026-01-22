import os
import typing
import warnings
from types import ModuleType
from warnings import warn
import rpy2.rinterface as rinterface
from . import conversion
from rpy2.robjects.functions import (SignatureTranslatedFunction,
from rpy2.robjects import Environment
from rpy2.robjects.packages_utils import (
import rpy2.robjects.help as rhelp
def run_withoutwarnings(*args, **kwargs):
    warn_i = _options().do_slot('names').index('warn')
    oldwarn = _options()[warn_i][0]
    _options(warn=-1)
    try:
        res = func(*args, **kwargs)
    except Exception as e:
        _options(warn=oldwarn)
        raise e
    _options(warn=oldwarn)
    return res