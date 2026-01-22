import contextvars
from functools import singledispatch
import os
from typing import Any
from typing import Optional
import typing
import warnings
from rpy2.rinterface_lib import _rinterface_capi
import rpy2.rinterface_lib.sexp
import rpy2.rinterface_lib.conversion
import rpy2.rinterface
def rclass_map_context(self, cls, d: typing.Dict[str, typing.Type]):
    return NameClassMapContext(self.rpy2py_nc_map[cls], d)