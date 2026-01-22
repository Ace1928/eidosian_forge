from fontTools.misc.fixedTools import (
from fontTools.misc.textTools import bytechr, byteord, bytesjoin, strjoin
from fontTools.pens.boundsPen import BoundsPen
import struct
import logging
def getNumRegions(self, vsindex=None):
    pd = self.private
    assert pd is not None
    if vsindex is not None:
        self._cur_vsindex = vsindex
    elif self._cur_vsindex is None:
        self._cur_vsindex = pd.vsindex if hasattr(pd, 'vsindex') else 0
    return pd.getNumRegions(self._cur_vsindex)