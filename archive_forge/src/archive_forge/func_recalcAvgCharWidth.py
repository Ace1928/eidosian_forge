from fontTools.misc import sstruct
from fontTools.misc.roundTools import otRound
from fontTools.misc.textTools import safeEval, num2binary, binary2num
from fontTools.ttLib.tables import DefaultTable
import bisect
import logging
def recalcAvgCharWidth(self, ttFont):
    """Recalculate xAvgCharWidth using metrics from ttFont's 'hmtx' table.

        Set it to 0 if the unlikely event 'hmtx' table is not found.
        """
    avg_width = 0
    hmtx = ttFont.get('hmtx')
    if hmtx is not None:
        widths = [width for width, _ in hmtx.metrics.values() if width > 0]
        if widths:
            avg_width = otRound(sum(widths) / len(widths))
    self.xAvgCharWidth = avg_width
    return avg_width