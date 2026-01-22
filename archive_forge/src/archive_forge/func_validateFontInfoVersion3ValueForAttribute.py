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
def validateFontInfoVersion3ValueForAttribute(attr, value):
    """
    This performs very basic validation of the value for attribute
    following the UFO 3 fontinfo.plist specification. The results
    of this should not be interpretted as *correct* for the font
    that they are part of. This merely indicates that the value
    is of the proper type and, where the specification defines
    a set range of possible values for an attribute, that the
    value is in the accepted range.
    """
    dataValidationDict = fontInfoAttributesVersion3ValueData[attr]
    valueType = dataValidationDict.get('type')
    validator = dataValidationDict.get('valueValidator')
    valueOptions = dataValidationDict.get('valueOptions')
    if valueOptions is not None:
        isValidValue = validator(value, valueOptions)
    elif validator == genericTypeValidator:
        isValidValue = validator(value, valueType)
    else:
        isValidValue = validator(value)
    return isValidValue