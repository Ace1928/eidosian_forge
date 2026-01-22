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
def readGlyph(self, glyphName, glyphObject=None, pointPen=None, validate=None):
    """
        Read a .glif file for 'glyphName' from the glyph set. The
        'glyphObject' argument can be any kind of object (even None);
        the readGlyph() method will attempt to set the following
        attributes on it:

        width
                the advance width of the glyph
        height
                the advance height of the glyph
        unicodes
                a list of unicode values for this glyph
        note
                a string
        lib
                a dictionary containing custom data
        image
                a dictionary containing image data
        guidelines
                a list of guideline data dictionaries
        anchors
                a list of anchor data dictionaries

        All attributes are optional, in two ways:

        1) An attribute *won't* be set if the .glif file doesn't
           contain data for it. 'glyphObject' will have to deal
           with default values itself.
        2) If setting the attribute fails with an AttributeError
           (for example if the 'glyphObject' attribute is read-
           only), readGlyph() will not propagate that exception,
           but ignore that attribute.

        To retrieve outline information, you need to pass an object
        conforming to the PointPen protocol as the 'pointPen' argument.
        This argument may be None if you don't need the outline data.

        readGlyph() will raise KeyError if the glyph is not present in
        the glyph set.

        ``validate`` will validate the data, by default it is set to the
        class's ``validateRead`` value, can be overridden.
        """
    if validate is None:
        validate = self._validateRead
    text = self.getGLIF(glyphName)
    try:
        tree = _glifTreeFromString(text)
        formatVersions = GLIFFormatVersion.supported_versions(self.ufoFormatVersionTuple)
        _readGlyphFromTree(tree, glyphObject, pointPen, formatVersions=formatVersions, validate=validate)
    except GlifLibError as glifLibError:
        fileName = self.contents[glyphName]
        try:
            glifLocation = f"'{self.fs.getsyspath(fileName)}'"
        except fs.errors.NoSysPath:
            glifLocation = f"'{fileName}' from '{str(self.fs)}'"
        glifLibError._add_note(f"The issue is in glyph '{glyphName}', located in {glifLocation}.")
        raise