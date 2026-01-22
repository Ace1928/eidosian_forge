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
def readFeatures(self):
    """
        Read features.fea. Return a string.
        The returned string is empty if the file is missing.
        """
    try:
        with self.fs.open(FEATURES_FILENAME, 'r', encoding='utf-8') as f:
            return f.read()
    except fs.errors.ResourceNotFound:
        return ''