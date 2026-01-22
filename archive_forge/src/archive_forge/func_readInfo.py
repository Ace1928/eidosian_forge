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
def readInfo(self, info, validate=None):
    """
        Read fontinfo.plist. It requires an object that allows
        setting attributes with names that follow the fontinfo.plist
        version 3 specification. This will write the attributes
        defined in the file into the object.

        ``validate`` will validate the read data, by default it is set to the
        class's validate value, can be overridden.
        """
    if validate is None:
        validate = self._validate
    infoDict = self._readInfo(validate)
    infoDataToSet = {}
    if self._formatVersion == UFOFormatVersion.FORMAT_1_0:
        for attr in fontInfoAttributesVersion1:
            value = infoDict.get(attr)
            if value is not None:
                infoDataToSet[attr] = value
        infoDataToSet = _convertFontInfoDataVersion1ToVersion2(infoDataToSet)
        infoDataToSet = _convertFontInfoDataVersion2ToVersion3(infoDataToSet)
    elif self._formatVersion == UFOFormatVersion.FORMAT_2_0:
        for attr, dataValidationDict in list(fontInfoAttributesVersion2ValueData.items()):
            value = infoDict.get(attr)
            if value is None:
                continue
            infoDataToSet[attr] = value
        infoDataToSet = _convertFontInfoDataVersion2ToVersion3(infoDataToSet)
    elif self._formatVersion.major == UFOFormatVersion.FORMAT_3_0.major:
        for attr, dataValidationDict in list(fontInfoAttributesVersion3ValueData.items()):
            value = infoDict.get(attr)
            if value is None:
                continue
            infoDataToSet[attr] = value
    else:
        raise NotImplementedError(self._formatVersion)
    if validate:
        infoDataToSet = validateInfoVersion3Data(infoDataToSet)
    for attr, value in list(infoDataToSet.items()):
        try:
            setattr(info, attr, value)
        except AttributeError:
            raise UFOLibError('The supplied info object does not support setting a necessary attribute (%s).' % attr)