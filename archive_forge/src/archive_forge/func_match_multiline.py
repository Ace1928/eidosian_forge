import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
def match_multiline(self, text, delimiter, in_state, style):
    """Do highlighting of multi-line strings. ``delimiter`` should be a
        ``QRegExp`` for triple-single-quotes or triple-double-quotes, and
        ``in_state`` should be a unique integer to represent the corresponding
        state changes when inside those strings. Returns True if we're still
        inside a multi-line string when this function is finished.
        """
    if self.previousBlockState() == in_state:
        start = 0
        add = 0
    else:
        start = delimiter.indexIn(text)
        add = delimiter.matchedLength()
    while start >= 0:
        end = delimiter.indexIn(text, start + add)
        if end >= add:
            length = end - start + add + delimiter.matchedLength()
            self.setCurrentBlockState(0)
        else:
            self.setCurrentBlockState(in_state)
            length = len(text) - start + add
        self.setFormat(start, length, self.styles[style])
        start = delimiter.indexIn(text, start + length)
    if self.currentBlockState() == in_state:
        return True
    else:
        return False