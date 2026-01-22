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
def makeUFOPath(path):
    """
    Return a .ufo pathname.

    >>> makeUFOPath("directory/something.ext") == (
    ... 	os.path.join('directory', 'something.ufo'))
    True
    >>> makeUFOPath("directory/something.another.thing.ext") == (
    ... 	os.path.join('directory', 'something.another.thing.ufo'))
    True
    """
    dir, name = os.path.split(path)
    name = '.'.join(['.'.join(name.split('.')[:-1]), 'ufo'])
    return os.path.join(dir, name)