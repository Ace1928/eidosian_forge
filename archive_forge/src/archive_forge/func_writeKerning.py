import os
from copy import deepcopy
from os import fsdecode
import logging
import zipfile
import enum
from collections import OrderedDict
import fs
import fs.base
import fs.subfs
import fs.errors
import fs.copy
import fs.osfs
import fs.zipfs
import fs.tempfs
import fs.tools
from fontTools.misc import plistlib
from fontTools.ufoLib.validators import *
from fontTools.ufoLib.filenames import userNameToFileName
from fontTools.ufoLib.converters import convertUFO1OrUFO2KerningToUFO3Kerning
from fontTools.ufoLib.errors import UFOLibError
from fontTools.ufoLib.utils import numberTypes, _VersionTupleEnumMixin
def writeKerning(self, kerning, validate=None):
    """
        Write kerning.plist. This method requires a
        dict of kerning pairs as an argument.

        This performs basic structural validation of the kerning,
        but it does not check for compliance with the spec in
        regards to conflicting pairs. The assumption is that the
        kerning data being passed is standards compliant.

        ``validate`` will validate the data, by default it is set to the
        class's validate value, can be overridden.
        """
    if validate is None:
        validate = self._validate
    if validate:
        invalidFormatMessage = 'The kerning is not properly formatted.'
        if not isDictEnough(kerning):
            raise UFOLibError(invalidFormatMessage)
        for pair, value in list(kerning.items()):
            if not isinstance(pair, (list, tuple)):
                raise UFOLibError(invalidFormatMessage)
            if not len(pair) == 2:
                raise UFOLibError(invalidFormatMessage)
            if not isinstance(pair[0], str):
                raise UFOLibError(invalidFormatMessage)
            if not isinstance(pair[1], str):
                raise UFOLibError(invalidFormatMessage)
            if not isinstance(value, numberTypes):
                raise UFOLibError(invalidFormatMessage)
    if self._formatVersion < UFOFormatVersion.FORMAT_3_0 and self._downConversionKerningData is not None:
        remap = self._downConversionKerningData['groupRenameMap']
        remappedKerning = {}
        for (side1, side2), value in list(kerning.items()):
            side1 = remap.get(side1, side1)
            side2 = remap.get(side2, side2)
            remappedKerning[side1, side2] = value
        kerning = remappedKerning
    kerningDict = {}
    for left, right in kerning.keys():
        value = kerning[left, right]
        if left not in kerningDict:
            kerningDict[left] = {}
        kerningDict[left][right] = value
    if kerningDict:
        self._writePlist(KERNING_FILENAME, kerningDict)
    elif self._havePreviousFile:
        self.removePath(KERNING_FILENAME, force=True, removeEmptyParents=False)