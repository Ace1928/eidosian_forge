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
def removeNames(self, nameID=None, platformID=None, platEncID=None, langID=None):
    """Remove any name records identified by the given combination of 'nameID',
        'platformID', 'platEncID' and 'langID'.
        """
    args = {argName: argValue for argName, argValue in (('nameID', nameID), ('platformID', platformID), ('platEncID', platEncID), ('langID', langID)) if argValue is not None}
    if not args:
        return
    self.names = [rec for rec in self.names if any((argValue != getattr(rec, argName) for argName, argValue in args.items()))]