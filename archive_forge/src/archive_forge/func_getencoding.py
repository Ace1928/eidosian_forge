import sys
import encodings
import encodings.aliases
import re
import _collections_abc
from builtins import str as _builtin_str
import functools
def getencoding():
    if hasattr(sys, 'getandroidapilevel'):
        return 'utf-8'
    encoding = getdefaultlocale()[1]
    if encoding is None:
        encoding = 'utf-8'
    return encoding