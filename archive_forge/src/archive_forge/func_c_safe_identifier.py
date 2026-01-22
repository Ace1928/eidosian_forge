from __future__ import absolute_import
import re
import copy
import operator
from ..Utils import try_finally_contextmanager
from .Errors import warning, error, InternalError, performance_hint
from .StringEncoding import EncodedString
from . import Options, Naming
from . import PyrexTypes
from .PyrexTypes import py_object_type, unspecified_type
from .TypeSlots import (
from . import Future
from . import Code
def c_safe_identifier(cname):
    if cname[:2] == '__' and (not (cname.startswith(Naming.pyrex_prefix) or cname in ('__weakref__', '__dict__'))) or cname in iso_c99_keywords:
        cname = Naming.pyrex_prefix + cname
    return cname