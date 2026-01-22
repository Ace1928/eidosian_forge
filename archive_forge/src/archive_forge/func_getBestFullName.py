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
def getBestFullName(self):
    for nameIDs in ((21, 22), (16, 17), (1, 2), (4,), (6,)):
        if len(nameIDs) == 2:
            name_fam = self.getDebugName(nameIDs[0])
            name_subfam = self.getDebugName(nameIDs[1])
            if None in [name_fam, name_subfam]:
                continue
            name = f'{name_fam} {name_subfam}'
            if name_subfam.lower() == 'regular':
                name = f'{name_fam}'
            return name
        else:
            name = self.getDebugName(nameIDs[0])
            if name is not None:
                return name
    return None