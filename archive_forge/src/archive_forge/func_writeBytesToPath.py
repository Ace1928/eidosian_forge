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
def writeBytesToPath(self, path, data):
    """
        Write bytes to a path relative to the UFO filesystem's root.
        If writing to an existing UFO, check to see if data matches the data
        that is already in the file at path; if so, the file is not rewritten
        so that the modification date is preserved.
        If needed, the directory tree for the given path will be built.
        """
    path = fsdecode(path)
    if self._havePreviousFile:
        if self.fs.isfile(path) and data == self.fs.readbytes(path):
            return
    try:
        self.fs.writebytes(path, data)
    except fs.errors.FileExpected:
        raise UFOLibError("A directory exists at '%s'" % path)
    except fs.errors.ResourceNotFound:
        self.fs.makedirs(fs.path.dirname(path), recreate=True)
        self.fs.writebytes(path, data)