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
def removePath(self, path, force=False, removeEmptyParents=True):
    """
        Remove the file (or directory) at path. The path
        must be relative to the UFO.
        Raises UFOLibError if the path doesn't exist.
        If force=True, ignore non-existent paths.
        If the directory where 'path' is located becomes empty, it will
        be automatically removed, unless 'removeEmptyParents' is False.
        """
    path = fsdecode(path)
    try:
        self.fs.remove(path)
    except fs.errors.FileExpected:
        self.fs.removetree(path)
    except fs.errors.ResourceNotFound:
        if not force:
            raise UFOLibError(f"'{path}' does not exist on {self.fs}")
    if removeEmptyParents:
        parent = fs.path.dirname(path)
        if parent:
            fs.tools.remove_empty(self.fs, parent)