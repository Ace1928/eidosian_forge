from . import model
from .commontypes import COMMON_TYPES, resolve_common_type
from .error import FFIError, CDefError
import weakref, re, sys
def convert_pycparser_error(self, e, csource):
    line = self._convert_pycparser_error(e, csource)
    msg = str(e)
    if line:
        msg = 'cannot parse "%s"\n%s' % (line.strip(), msg)
    else:
        msg = 'parse error\n%s' % (msg,)
    raise CDefError(msg)