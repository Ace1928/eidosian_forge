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
def punycodify_name(cname, mangle_with=None):
    try:
        cname.encode('ascii')
    except UnicodeEncodeError:
        cname = cname.encode('punycode').replace(b'-', b'_').decode('ascii')
        if mangle_with:
            cname = '%s_%s' % (mangle_with, cname)
        elif cname.startswith(Naming.pyrex_prefix):
            cname = cname.replace(Naming.pyrex_prefix, Naming.pyunicode_identifier_prefix, 1)
    return cname