from __future__ import absolute_import, division, unicode_literals
import re
import warnings
from .constants import DataLossWarning
def charStringToList(chars):
    charRanges = [item.strip() for item in chars.split(' | ')]
    rv = []
    for item in charRanges:
        foundMatch = False
        for regexp in (reChar, reCharRange):
            match = regexp.match(item)
            if match is not None:
                rv.append([hexToInt(item) for item in match.groups()])
                if len(rv[-1]) == 1:
                    rv[-1] = rv[-1] * 2
                foundMatch = True
                break
        if not foundMatch:
            assert len(item) == 1
            rv.append([ord(item)] * 2)
    rv = normaliseCharList(rv)
    return rv