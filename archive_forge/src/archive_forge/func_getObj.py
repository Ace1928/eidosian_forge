import os
import re
import sys
from codecs import BOM_UTF8, BOM_UTF16, BOM_UTF16_BE, BOM_UTF16_LE
import six
from ._version import __version__
def getObj(s):
    global compiler
    if compiler is None:
        import compiler
    s = 'a=' + s
    p = compiler.parse(s)
    return p.getChildren()[1].getChildren()[0].getChildren()[1]