from __future__ import annotations
import logging
import enum
from warnings import warn
from collections import OrderedDict
import fs
import fs.base
import fs.errors
import fs.osfs
import fs.path
from fontTools.misc.textTools import tobytes
from fontTools.misc import plistlib
from fontTools.pens.pointPen import AbstractPointPen, PointToSegmentPen
from fontTools.ufoLib.errors import GlifLibError
from fontTools.ufoLib.filenames import userNameToFileName
from fontTools.ufoLib.validators import (
from fontTools.misc import etree
from fontTools.ufoLib import _UFOBaseIO, UFOFormatVersion
from fontTools.ufoLib.utils import numberTypes, _VersionTupleEnumMixin
def getGLIFModificationTime(self, glyphName):
    """
        Returns the modification time for the GLIF file with 'glyphName', as
        a floating point number giving the number of seconds since the epoch.
        Return None if the associated file does not exist or the underlying
        filesystem does not support getting modified times.
        Raises KeyError if the glyphName is not in contents.plist.
        """
    fileName = self.contents[glyphName]
    return self.getFileModificationTime(fileName)