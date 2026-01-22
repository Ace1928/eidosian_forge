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
def writeImage(self, fileName, data, validate=None):
    """
        Write data to fileName in the images directory.
        The data must be a valid PNG.
        """
    if validate is None:
        validate = self._validate
    if self._formatVersion < UFOFormatVersion.FORMAT_3_0:
        raise UFOLibError(f'Images are not allowed in UFO {self._formatVersion.major}.')
    fileName = fsdecode(fileName)
    if validate:
        valid, error = pngValidator(data=data)
        if not valid:
            raise UFOLibError(error)
    self.writeBytesToPath(f'{IMAGES_DIRNAME}/{fileName}', data)