from fontTools.misc.fixedTools import floatToFixedToFloat, floatToFixed
from fontTools.misc.roundTools import otRound
from fontTools.pens.boundsPen import BoundsPen
from fontTools.ttLib import TTFont, newTable
from fontTools.ttLib.tables import ttProgram
from fontTools.ttLib.tables._g_l_y_f import (
from fontTools.varLib.models import (
from fontTools.varLib.merger import MutatorMerger
from fontTools.varLib.varStore import VarStoreInstancer
from fontTools.varLib.mvar import MVAR_ENTRIES
from fontTools.varLib.iup import iup_delta
import fontTools.subset.cff
import os.path
import logging
from io import BytesIO
If any argument in a BlueValues list is a blend list,
                then they all are. The first value of each list is an
                absolute value. The delta tuples are calculated from
                relative master values, hence we need to append all the
                deltas to date to each successive absolute value.