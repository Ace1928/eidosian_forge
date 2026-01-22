import os
from collections import namedtuple
import re
import sqlite3
import typing
import warnings
import rpy2.rinterface as rinterface
from rpy2.rinterface import StrSexpVector
from rpy2.robjects.packages_utils import (get_packagepath,
from collections import OrderedDict
def quiet_require(name: str, lib_loc: typing.Optional[str]=None) -> bool:
    """ Load an R package /quietly/ (suppressing messages to the console). """
    if lib_loc is None:
        lib_loc = 'NULL'
    expr_txt = 'suppressPackageStartupMessages(base::require(%s, lib.loc=%s))' % (name, lib_loc)
    expr = rinterface.parse(expr_txt)
    ok = _eval(expr)
    return ok