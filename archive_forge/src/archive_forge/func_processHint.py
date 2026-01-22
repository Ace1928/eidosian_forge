from fontTools.misc import psCharStrings
from fontTools import ttLib
from fontTools.pens.basePen import NullPen
from fontTools.misc.roundTools import otRound
from fontTools.misc.loggingTools import deprecateFunction
from fontTools.subset.util import _add_method, _uniq_sort
def processHint(self, index):
    cs = self.callingStack[-1]
    hints = cs._hints
    hints.has_hint = True
    hints.last_hint = index
    hints.last_checked = index