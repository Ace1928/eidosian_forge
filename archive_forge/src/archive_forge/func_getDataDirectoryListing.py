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
def getDataDirectoryListing(self):
    """
        Returns a list of all files in the data directory.
        The returned paths will be relative to the UFO.
        This will not list directory names, only file names.
        Thus, empty directories will be skipped.
        """
    try:
        self._dataFS = self.fs.opendir(DATA_DIRNAME)
    except fs.errors.ResourceNotFound:
        return []
    except fs.errors.DirectoryExpected:
        raise UFOLibError('The UFO contains a "data" file instead of a directory.')
    try:
        return [p.lstrip('/') for p in self._dataFS.walk.files()]
    except fs.errors.ResourceError:
        return []