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
def getImageDirectoryListing(self, validate=None):
    """
        Returns a list of all image file names in
        the images directory. Each of the images will
        have been verified to have the PNG signature.

        ``validate`` will validate the data, by default it is set to the
        class's validate value, can be overridden.
        """
    if self._formatVersion < UFOFormatVersion.FORMAT_3_0:
        return []
    if validate is None:
        validate = self._validate
    try:
        self._imagesFS = imagesFS = self.fs.opendir(IMAGES_DIRNAME)
    except fs.errors.ResourceNotFound:
        return []
    except fs.errors.DirectoryExpected:
        raise UFOLibError('The UFO contains an "images" file instead of a directory.')
    result = []
    for path in imagesFS.scandir('/'):
        if path.is_dir:
            continue
        if validate:
            with imagesFS.openbin(path.name) as fp:
                valid, error = pngValidator(fileObj=fp)
            if valid:
                result.append(path.name)
        else:
            result.append(path.name)
    return result