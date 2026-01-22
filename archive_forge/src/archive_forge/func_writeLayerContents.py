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
def writeLayerContents(self, layerOrder=None, validate=None):
    """
        Write the layercontents.plist file. This method  *must* be called
        after all glyph sets have been written.
        """
    if validate is None:
        validate = self._validate
    if self._formatVersion < UFOFormatVersion.FORMAT_3_0:
        return
    if layerOrder is not None:
        newOrder = []
        for layerName in layerOrder:
            if layerName is None:
                layerName = DEFAULT_LAYER_NAME
            newOrder.append(layerName)
        layerOrder = newOrder
    else:
        layerOrder = list(self.layerContents.keys())
    if validate and set(layerOrder) != set(self.layerContents.keys()):
        raise UFOLibError('The layer order content does not match the glyph sets that have been created.')
    layerContents = [(layerName, self.layerContents[layerName]) for layerName in layerOrder]
    self._writePlist(LAYERCONTENTS_FILENAME, layerContents)