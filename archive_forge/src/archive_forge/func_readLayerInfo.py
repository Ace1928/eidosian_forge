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
def readLayerInfo(self, info, validateRead=None):
    """
        ``validateRead`` will validate the data, by default it is set to the
        class's ``validateRead`` value, can be overridden.
        """
    if validateRead is None:
        validateRead = self._validateRead
    infoDict = self._getPlist(LAYERINFO_FILENAME, {})
    if validateRead:
        if not isinstance(infoDict, dict):
            raise GlifLibError('layerinfo.plist is not properly formatted.')
        infoDict = validateLayerInfoVersion3Data(infoDict)
    for attr, value in infoDict.items():
        try:
            setattr(info, attr, value)
        except AttributeError:
            raise GlifLibError('The supplied layer info object does not support setting a necessary attribute (%s).' % attr)