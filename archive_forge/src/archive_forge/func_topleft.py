import doctest
import collections
@topleft.setter
def topleft(self, value):
    if self._readOnly:
        raise PyRectException('Rect object is read-only')
    _checkForTwoIntOrFloatTuple(value)
    newLeft, newTop = value
    if newLeft != self._left or newTop != self._top:
        originalLeft = self._left
        originalTop = self._top
        if self._enableFloat:
            self._left = newLeft
            self._top = newTop
        else:
            self._left = int(newLeft)
            self._top = int(newTop)
        self.callOnChange(originalLeft, originalTop, self._width, self._height)