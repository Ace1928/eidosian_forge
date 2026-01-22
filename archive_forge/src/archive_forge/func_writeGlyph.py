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
def writeGlyph(self, glyphName, glyphObject=None, drawPointsFunc=None, formatVersion=None, validate=None):
    """
        Write a .glif file for 'glyphName' to the glyph set. The
        'glyphObject' argument can be any kind of object (even None);
        the writeGlyph() method will attempt to get the following
        attributes from it:

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

        All attributes are optional: if 'glyphObject' doesn't
        have the attribute, it will simply be skipped.

        To write outline data to the .glif file, writeGlyph() needs
        a function (any callable object actually) that will take one
        argument: an object that conforms to the PointPen protocol.
        The function will be called by writeGlyph(); it has to call the
        proper PointPen methods to transfer the outline to the .glif file.

        The GLIF format version will be chosen based on the ufoFormatVersion
        passed during the creation of this object. If a particular format
        version is desired, it can be passed with the formatVersion argument.
        The formatVersion argument accepts either a tuple of integers for
        (major, minor), or a single integer for the major digit only (with
        minor digit implied as 0).

        An UnsupportedGLIFFormat exception is raised if the requested GLIF
        formatVersion is not supported.

        ``validate`` will validate the data, by default it is set to the
        class's ``validateWrite`` value, can be overridden.
        """
    if formatVersion is None:
        formatVersion = GLIFFormatVersion.default(self.ufoFormatVersionTuple)
    else:
        try:
            formatVersion = GLIFFormatVersion(formatVersion)
        except ValueError as e:
            from fontTools.ufoLib.errors import UnsupportedGLIFFormat
            raise UnsupportedGLIFFormat(f'Unsupported GLIF format version: {formatVersion!r}') from e
    if formatVersion not in GLIFFormatVersion.supported_versions(self.ufoFormatVersionTuple):
        from fontTools.ufoLib.errors import UnsupportedGLIFFormat
        raise UnsupportedGLIFFormat(f'Unsupported GLIF format version ({formatVersion!s}) for UFO format version {self.ufoFormatVersionTuple!s}.')
    if validate is None:
        validate = self._validateWrite
    fileName = self.contents.get(glyphName)
    if fileName is None:
        if self._existingFileNames is None:
            self._existingFileNames = {fileName.lower() for fileName in self.contents.values()}
        fileName = self.glyphNameToFileName(glyphName, self._existingFileNames)
        self.contents[glyphName] = fileName
        self._existingFileNames.add(fileName.lower())
        if self._reverseContents is not None:
            self._reverseContents[fileName.lower()] = glyphName
    data = _writeGlyphToBytes(glyphName, glyphObject, drawPointsFunc, formatVersion=formatVersion, validate=validate)
    if self._havePreviousFile and self.fs.exists(fileName) and (data == self.fs.readbytes(fileName)):
        return
    self.fs.writebytes(fileName, data)