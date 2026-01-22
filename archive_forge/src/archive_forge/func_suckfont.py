from fontTools.misc.textTools import bytechr, byteord, bytesjoin, tobytes, tostr
from fontTools.misc import eexec
from .psOperators import (
import re
from collections.abc import Callable
from string import whitespace
import logging
def suckfont(data, encoding='ascii'):
    m = re.search(b'/FontName\\s+/([^ \\t\\n\\r]+)\\s+def', data)
    if m:
        fontName = m.group(1)
        fontName = fontName.decode()
    else:
        fontName = None
    interpreter = PSInterpreter(encoding=encoding)
    interpreter.interpret(b'/Helvetica 4 dict dup /Encoding StandardEncoding put definefont pop')
    interpreter.interpret(data)
    fontdir = interpreter.dictstack[0]['FontDirectory'].value
    if fontName in fontdir:
        rawfont = fontdir[fontName]
    else:
        fontNames = list(fontdir.keys())
        if len(fontNames) > 1:
            fontNames.remove('Helvetica')
        fontNames.sort()
        rawfont = fontdir[fontNames[0]]
    interpreter.close()
    return unpack_item(rawfont)