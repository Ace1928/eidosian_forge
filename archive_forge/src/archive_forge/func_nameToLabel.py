from __future__ import annotations
import errno
import os
import sys
import warnings
from typing import AnyStr
from collections import OrderedDict
from typing import (
from incremental import Version
from twisted.python.deprecate import deprecatedModuleAttribute
def nameToLabel(mname):
    """
    Convert a string like a variable name into a slightly more human-friendly
    string with spaces and capitalized letters.

    @type mname: C{str}
    @param mname: The name to convert to a label.  This must be a string
    which could be used as a Python identifier.  Strings which do not take
    this form will result in unpredictable behavior.

    @rtype: C{str}
    """
    labelList = []
    word = ''
    lastWasUpper = False
    for letter in mname:
        if letter.isupper() == lastWasUpper:
            word += letter
        elif lastWasUpper:
            if len(word) == 1:
                word += letter
            else:
                lastWord = word[:-1]
                firstLetter = word[-1]
                labelList.append(lastWord)
                word = firstLetter + letter
        else:
            labelList.append(word)
            word = letter
        lastWasUpper = letter.isupper()
    if labelList:
        labelList[0] = labelList[0].capitalize()
    else:
        return mname.capitalize()
    labelList.append(word)
    return ' '.join(labelList)