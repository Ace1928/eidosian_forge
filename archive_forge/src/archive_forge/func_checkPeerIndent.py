import string
from weakref import ref as wkref
import copy
import sys
import warnings
import re
import sre_constants
import collections
def checkPeerIndent(s, l, t):
    if l >= len(s):
        return
    curCol = col(l, s)
    if curCol != indentStack[-1]:
        if curCol > indentStack[-1]:
            raise ParseFatalException(s, l, 'illegal nesting')
        raise ParseException(s, l, 'not a peer entry')