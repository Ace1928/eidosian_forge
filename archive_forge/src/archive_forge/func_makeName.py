from fontTools.misc import sstruct
from fontTools.misc.textTools import (
from fontTools.misc.encodingTools import getEncoding
from fontTools.ttLib import newTable
from fontTools.ttLib.ttVisitor import TTVisitor
from fontTools import ttLib
import fontTools.ttLib.tables.otTables as otTables
from fontTools.ttLib.tables import C_P_A_L_
from . import DefaultTable
import struct
import logging
def makeName(string, nameID, platformID, platEncID, langID):
    name = NameRecord()
    name.string, name.nameID, name.platformID, name.platEncID, name.langID = (string, nameID, platformID, platEncID, langID)
    return name