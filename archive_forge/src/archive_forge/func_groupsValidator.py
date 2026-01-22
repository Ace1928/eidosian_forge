import calendar
from io import open
import fs.base
import fs.osfs
from collections.abc import Mapping
from fontTools.ufoLib.utils import numberTypes
def groupsValidator(value):
    """
    Check the validity of the groups.
    Version 3+ (though it's backwards compatible with UFO 1 and UFO 2).

    >>> groups = {"A" : ["A", "A"], "A2" : ["A"]}
    >>> groupsValidator(groups)
    (True, None)

    >>> groups = {"" : ["A"]}
    >>> valid, msg = groupsValidator(groups)
    >>> valid
    False
    >>> print(msg)
    A group has an empty name.

    >>> groups = {"public.awesome" : ["A"]}
    >>> groupsValidator(groups)
    (True, None)

    >>> groups = {"public.kern1." : ["A"]}
    >>> valid, msg = groupsValidator(groups)
    >>> valid
    False
    >>> print(msg)
    The group data contains a kerning group with an incomplete name.
    >>> groups = {"public.kern2." : ["A"]}
    >>> valid, msg = groupsValidator(groups)
    >>> valid
    False
    >>> print(msg)
    The group data contains a kerning group with an incomplete name.

    >>> groups = {"public.kern1.A" : ["A"], "public.kern2.A" : ["A"]}
    >>> groupsValidator(groups)
    (True, None)

    >>> groups = {"public.kern1.A1" : ["A"], "public.kern1.A2" : ["A"]}
    >>> valid, msg = groupsValidator(groups)
    >>> valid
    False
    >>> print(msg)
    The glyph "A" occurs in too many kerning groups.
    """
    bogusFormatMessage = 'The group data is not in the correct format.'
    if not isDictEnough(value):
        return (False, bogusFormatMessage)
    firstSideMapping = {}
    secondSideMapping = {}
    for groupName, glyphList in value.items():
        if not isinstance(groupName, str):
            return (False, bogusFormatMessage)
        if not isinstance(glyphList, (list, tuple)):
            return (False, bogusFormatMessage)
        if not groupName:
            return (False, 'A group has an empty name.')
        if groupName.startswith('public.'):
            if not groupName.startswith('public.kern1.') and (not groupName.startswith('public.kern2.')):
                continue
            elif len('public.kernN.') == len(groupName):
                return (False, 'The group data contains a kerning group with an incomplete name.')
            if groupName.startswith('public.kern1.'):
                d = firstSideMapping
            else:
                d = secondSideMapping
            for glyphName in glyphList:
                if not isinstance(glyphName, str):
                    return (False, 'The group data %s contains an invalid member.' % groupName)
                if glyphName in d:
                    return (False, 'The glyph "%s" occurs in too many kerning groups.' % glyphName)
                d[glyphName] = groupName
    return (True, None)