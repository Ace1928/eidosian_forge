import traceback
from io import StringIO
from java.lang import StringBuffer  # @UnresolvedImport
from java.lang import String  # @UnresolvedImport
import java.lang  # @UnresolvedImport
import sys
from _pydev_bundle._pydev_tipper_common import do_find
from org.python.core import PyReflectedFunction  # @UnresolvedImport
from org.python import core  # @UnresolvedImport
from org.python.core import PyClass  # @UnresolvedImport
import java.util
def format_arg(arg):
    """formats an argument to be shown
    """
    s = str(arg)
    dot = s.rfind('.')
    if dot >= 0:
        s = s[dot + 1:]
    s = s.replace(';', '')
    s = s.replace('[]', 'Array')
    if len(s) > 0:
        c = s[0].lower()
        s = c + s[1:]
    return s